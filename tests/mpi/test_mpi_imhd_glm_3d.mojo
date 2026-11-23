# Multi-rank regression: 3D GLM-MHD, periodic, with HLLD, the limiter and
# the GLM damping source.
#
# 8 ranks cut the 10 x 8 x 6 mesh along all three axes, so every one of the
# six directions is exchanged, with unequal boxes along x. The field must
# be bitwise independent of the number of ranks.
#
# MPI_RANKS: 1 2 3 4 8

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *

comptime P: Int = 1
comptime GAMMA: Real = 5.0 / 3.0
comptime C_H: Real = 2.0
comptime C_P_SQ: Real = 0.4
comptime State = DGState[3, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=10, ny=8, nz=6,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    # A 3D Orszag-Tang-like vortex: smooth, fully three-dimensional and
    # steepening, so the limiter has work to do before the end.
    def initial(x: Real, y: Real, z: Real) -> Array[Real, MHD_GLM_NC]:
        var rho: Real = 25.0 / (36.0 * PI)
        var pp: Real = 5.0 / (12.0 * PI)
        var b0: Real = 0.28
        var ux = -sin(TWO_PI * y) * (Real(1.0) + Real(0.2) * sin(TWO_PI * z))
        var uy = sin(TWO_PI * x) * (Real(1.0) + Real(0.2) * cos(TWO_PI * z))
        var uz = Real(0.2) * sin(TWO_PI * (x + y))
        var bx = -b0 * sin(TWO_PI * y)
        var by = b0 * sin(Real(2.0) * TWO_PI * x)
        var bz = b0 * Real(0.3) * cos(TWO_PI * x)
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[IDX_RHO] = rho
        v[IDX_MX] = rho * ux
        v[IDX_MY] = rho * uy
        v[IDX_MZ] = rho * uz
        v[IDX_BX] = bx
        v[IDX_BY] = by
        v[IDX_BZ] = bz
        v[IDX_E] = (
            pp / (GAMMA - Real(1.0))
            + Real(0.5) * rho * (ux * ux + uy * uy + uz * uz)
            + Real(0.5) * (bx * bx + by * by + bz * bz)
        )
        return v^
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-10),
        min_pressure=Real(1e-10), cfl_max=Real(0.4),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[3, P, USE_HLLD=True](app)
    var stepper = SSPRK3[3, P, MHD_GLM_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[3, P, MHD_GLM_NC](
        ctx, mesh.num_elements, mesh.num_ghost, alpha_coeff=Real(1.0),
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), Real(0.15), compute_rhs, adjust, suggest_dt,
    )
    var mass_1 = mesh.integrate(q, IDX_RHO)
    var sum = mesh.checksum(q)
    if is_root():
        print("ranks =", num_ranks(), " mass drift =", mass_1 - mass_0)
        print("CHECKSUM", sum)
    if not (fabs(mass_1 - mass_0) < Real(1.0e-13)):
        raise Error("mass not conserved on the periodic box")
    if is_root():
        print("PASSED")
