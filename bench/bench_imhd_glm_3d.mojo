# 3D IMHD-GLM bench driver.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime C_H: Real = 4.0
comptime C_P_SQ: Real = 0.5
comptime DT: Real = 5.0e-4
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100
comptime State = DGState[3, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx,
        Real(0.0), TWO_PI,
        Real(0.0), TWO_PI,
        Real(0.0), TWO_PI,
        nx=NX, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = State(ctx, mesh.num_elements)

    var rho_0 = GAMMA * GAMMA
    var p_0: Real = GAMMA
    # 2D Orszag-Tang-like IC extruded along z.
    def initial(x: Real, y: Real, z: Real) {imm} -> Array[Real, MHD_GLM_NC]:
        var ux = -sin(y)
        var uy = sin(x)
        var Bx_v = -sin(y)
        var By_v = sin(Real(2.0) * x)
        var rho = rho_0
        var msq = rho * (ux * ux + uy * uy)
        var bsq = Bx_v * Bx_v + By_v * By_v
        var E_tot = (
            p_0 / (GAMMA - Real(1.0))
            + Real(0.5) * msq + Real(0.5) * bsq
        )
        return [
            rho, rho * ux, rho * uy, Real(0.0), E_tot,
            Bx_v, By_v, Real(0.0), Real(0.0),
        ]
    mesh.fill(q, initial)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, MHD_GLM_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    for _ in range(WARMUP):
        stepper.step(ctx, q, DT, compute_rhs, adjust)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, adjust)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var elapsed_s = Real(t1 - t0) / Real(1.0e9)
    var steps_per_s = Real(N_STEPS) / elapsed_s
    print(
        "BENCH config=imhd_glm_3d steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
    )
