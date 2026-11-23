# 2D MHD Orszag-Tang vortex with time-series VTU output and diagnostics.
#
# Upstream WARPXM convention: periodic [0, 2π]², ρ₀ = γ², p₀ = γ. GLM-MHD
# with Dedner cleaning, Moe-Rossmanith limiter, SSPRK3 with a CFL-driven
# time step. Each frame prints ∫ρ, ∫E and ∫ψ² so you can watch
# conservation hold and the cleaning energy stay bounded.
#
#     pixi run example imhd_orszag_tang_2d_vtu
#     # then open output/orszag_tang/orszag_tang.pvd in ParaView

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 64
comptime GAMMA: Real = 5.0 / 3.0
comptime C_H: Real = 4.0
comptime C_P_SQ: Real = 0.5
comptime T_FINAL: Real = 0.5
comptime NUM_FRAMES: Int = 11           # frames 0 .. 10 inclusive
comptime State = DGState[2, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), TWO_PI, Real(0.0), TWO_PI,
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real) -> Array[Real, MHD_GLM_NC]:
        var rho = GAMMA * GAMMA
        var ux = -sin(y)
        var uy = sin(x)
        var bx = -sin(y)
        var by = sin(Real(2.0) * x)
        var energy = (
            GAMMA / (GAMMA - Real(1.0))
            + Real(0.5) * rho * (ux * ux + uy * uy)
            + Real(0.5) * (bx * bx + by * by)
        )
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[IDX_RHO] = rho
        v[IDX_MX] = rho * ux
        v[IDX_MY] = rho * uy
        v[IDX_E] = energy
        v[IDX_BX] = bx
        v[IDX_BY] = by
        return v^
    mesh.fill(q, initial)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[2, P](app)
    var stepper = SSPRK3[2, P, MHD_GLM_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[2, P, MHD_GLM_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var fw = FrameWriter("orszag_tang")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    def psi_squared(v: Array[Real, MHD_GLM_NC]) -> Real:
        return v[IDX_PSI] * v[IDX_PSI]

    var mass_0 = mesh.integrate(q, IDX_RHO)
    var energy_0 = mesh.integrate(q, IDX_E)
    root_print("time   ∫ρ dV   ∫E dV   ∫ψ² dV   rel Δmass   rel ΔE")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_imhd_glm_frame(fw.next_path(t), mesh, q, GAMMA)
        var mass = mesh.integrate(q, IDX_RHO)
        var energy = mesh.integrate(q, IDX_E)
        root_print(
            t, mass, energy, mesh.integrate_fn(q, psi_squared),
            (mass - mass_0) / mass_0, (energy - energy_0) / energy_0,
        )
        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt, dt_interval=10,
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/orszag_tang.pvd")
