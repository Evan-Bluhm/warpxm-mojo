# 2D MHD Orszag-Tang vortex — Athena code normalization — with VTU
# time-series output. Direct visual comparison target for the canonical
# Athena reference figures at:
#   https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/
#
# The setup mirrors `~/GitHub/Athena-Cversion/src/prob/orszag-tang.c`:
#   * Domain [0, 1]² periodic; γ = 5/3
#   * d_0 = 25/(36π), p_0 = 5/(12π), v_0 = 1, B_0 = 1/√(4π)
#   * Vector potential A_z = (B_0/4π) cos(4π x) + (B_0/2π) cos(2π y),
#     so B_x = -B_0 sin(2π y), B_y = B_0 sin(4π x).
#
# Frames are dumped from t = 0 to t = 1 (the canonical Athena endpoint)
# at uniform intervals. Per frame the driver prints (∫ρ dV, ∫E dV, ∫ψ² dV)
# so the user can verify mass / energy conservation and that GLM
# divergence cleaning is keeping ∇·B errors damped.
#
# Build / run:
#     mkdir -p /tmp/orszag_tang_athena
#     pixi run mojo run -I src examples/imhd_orszag_tang_2d_athena_vtu.mojo
#     # Then open /tmp/orszag_tang_athena/orszag_tang_athena.pvd in ParaView.

from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.adjusters.positivity_mhd import PositivityMHD2D
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC, IDX_PSI
from warpxm.apps.imhd import (
    IDX_RHO,
    IDX_MX,
    IDX_MY,
    IDX_MZ,
    IDX_E,
    IDX_BX,
    IDX_BY,
    IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, PI, TWO_PI
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_imhd_2d import dump_imhd_glm_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_glm_2d import DGImhdGLMSolver2D
from warpxm.spatial.imhd_glm_source_2d import apply_glm_damping_2d
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 3
comptime NX: Int = 128
comptime NY: Int = 128
comptime GAMMA: Real = 5.0 / 3.0
comptime V0: Real = 1.0
comptime C_H: Real = 3.0
comptime C_P_SQ: Real = 0.18
comptime T_FINAL: Real = 4.0
comptime DT: Real = 2.5e-5
comptime NUM_FRAMES: Int = 41  # frames 0 .. 40 inclusive
comptime OUTPUT_DIR: String = "/tmp/orszag_tang_athena"
comptime BASENAME: String = "orszag_tang_athena"


def _integrate_field[
    F: Int
](var ctx: DeviceContext, mut q: DGState2D[P, MHD_GLM_NC], mesh: BlockMesh2D[P],) raises -> Real:
    """Integrate component `F` of `q` over the domain using GLL quadrature."""
    var P1 = NodalBasis1D[P].num_nodes
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var s: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                        s += ja * weights[in_] * weights[jn] * h[b + F]
    return s


def _integrate_squared_field[
    F: Int
](var ctx: DeviceContext, mut q: DGState2D[P, MHD_GLM_NC], mesh: BlockMesh2D[P],) raises -> Real:
    """Integrate `q[F]²` over the domain (useful for ψ² and energies)."""
    var P1 = NodalBasis1D[P].num_nodes
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var s: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                        var v = h[b + F]
                        s += ja * weights[in_] * weights[jn] * v * v
    return s


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx,
        Real(0.0),
        Real(1.0),
        Real(0.0),
        Real(1.0),
        nx=NX,
        ny=NY,
        periodic_x=True,
        periodic_y=True,
    )
    var q = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1

    var d0: Real = Real(25.0) / (Real(36.0) * PI)
    var p0: Real = Real(5.0) / (Real(12.0) * PI)
    var B0: Real = Real(1.0) / sqrt(Real(4.0) * PI)

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var ux = -V0 * sin(TWO_PI * y)
                    var uy = V0 * sin(TWO_PI * x)
                    var Bx_v = -B0 * sin(TWO_PI * y)
                    var By_v = B0 * sin(Real(2.0) * TWO_PI * x)
                    var msq = d0 * (ux * ux + uy * uy)
                    var bsq = Bx_v * Bx_v + By_v * By_v
                    var E_tot = p0 / (GAMMA - Real(1.0)) + Real(0.5) * msq + Real(0.5) * bsq
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                    host[b + IDX_RHO] = d0
                    host[b + IDX_MX] = d0 * ux
                    host[b + IDX_MY] = d0 * uy
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
                    host[b + IDX_BX] = Bx_v
                    host[b + IDX_BY] = By_v
                    host[b + IDX_BZ] = Real(0.0)
                    host[b + IDX_PSI] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = ImhdGLM(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
        c_h=C_H,
        c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var positivity = PositivityMHD2D[P, MHD_GLM_NC](
        num_elements=mesh.num_elements,
        gamma=GAMMA,
        rho_eps=Real(1.0e-10),
        p_eps=Real(1.0e-10),
    )
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)
    var mass_0 = _integrate_field[IDX_RHO](ctx, q, mesh)
    var e_0 = _integrate_field[IDX_E](ctx, q, mesh)

    print("Orszag-Tang vortex (Athena units) — domain [0,1]², P =", P, ", grid", NX, "x", NY)
    print("  d_0 =", d0, " p_0 =", p0, " B_0 =", B0)
    print("")
    print("time         ∫ρ dV            ∫E dV            ∫ψ² dV         rel Δmass   rel ΔE")
    print("-----------  ---------------- ---------------- ---------------- ----------  ----------")

    def compute_rhs(
        mut q_in: DGState2D[P, MHD_GLM_NC],
        mut q_out: DGState2D[P, MHD_GLM_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_2d[P](ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: DGState2D[P, MHD_GLM_NC]) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_imhd_glm_2d_frame[P, MHD_GLM_NC](fw.next_path(t), ctx, mesh, q, GAMMA)
        var mass = _integrate_field[IDX_RHO](ctx, q, mesh)
        var e_tot = _integrate_field[IDX_E](ctx, q, mesh)
        var psi2 = _integrate_squared_field[IDX_PSI](ctx, q, mesh)
        var dm = (mass - mass_0) / mass_0
        var dE = (e_tot - e_0) / e_0
        print(t, " ", mass, " ", e_tot, " ", psi2, " ", dm, " ", dE)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step(ctx, q, dt_use, compute_rhs, adjust)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
