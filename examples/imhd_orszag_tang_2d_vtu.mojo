# 2D MHD Orszag-Tang vortex with time-series VTU output and integrated
# diagnostics — full-stack demonstration of the framework.
#
# This driver wires together every major piece of the 2D MHD pipeline:
#   * `BlockMesh2D` on a periodic [0, 2π]² domain
#   * `ImhdGLM` for ideal MHD with Dedner divergence cleaning (9 components)
#   * `DGImhdGLMSolver2D` for the volume + face flux
#   * `MoeRossmanith2D[P, 9]` for shock capturing
#   * `SSPRK3_IMHD_GLM_Limited_2D` for time stepping
#   * `FrameWriter` + `dump_imhd_glm_2d_frame` for ParaView output
#
# Per frame it prints (∫ρ dV, ∫E dV, ∫ψ² dV) so the user can watch
# conservation hold and divergence-cleaning energy decay in real time.
#
# Build / run:
#     mkdir -p /tmp/orszag_tang
#     pixi run mojo run -I src examples/imhd_orszag_tang_2d_vtu.mojo
#     # Then open /tmp/orszag_tang/orszag_tang.pvd in ParaView.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC, IDX_PSI
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_imhd_2d import dump_imhd_glm_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_glm_2d import DGImhdGLMSolver2D
from warpxm.spatial.imhd_glm_source_2d import apply_glm_damping_2d
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 64
comptime GAMMA: Real = 5.0 / 3.0
comptime C_H: Real = 4.0
comptime C_P_SQ: Real = 0.5
comptime T_FINAL: Real = 0.5
comptime DT: Real = 1.0e-3
comptime NUM_FRAMES: Int = 11           # frames 0 .. 10 inclusive
comptime OUTPUT_DIR: String = "/tmp/orszag_tang"
comptime BASENAME: String = "orszag_tang"


def _integrate_field[F: Int](
    var ctx: DeviceContext,
    mut q: DGState2D[P, MHD_GLM_NC],
    mesh: BlockMesh2D[P],
) raises -> Real:
    """Integrate component `F` of `q` over the domain using GLL quadrature."""
    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
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


def _integrate_squared_field[F: Int](
    var ctx: DeviceContext,
    mut q: DGState2D[P, MHD_GLM_NC],
    mesh: BlockMesh2D[P],
) raises -> Real:
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
        ctx, Real(0.0), TWO_PI, Real(0.0), TWO_PI,
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1
    var rho_0 = GAMMA * GAMMA
    var p_0: Real = GAMMA

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
                    var ux = -sin(y)
                    var uy = sin(x)
                    var Bx_v = -sin(y)
                    var By_v = sin(Real(2.0) * x)
                    var rho = rho_0
                    var msq = rho * (ux * ux + uy * uy)
                    var bsq = Bx_v * Bx_v + By_v * By_v
                    var E_tot = p_0 / (GAMMA - Real(1.0)) + Real(0.5) * msq + Real(0.5) * bsq
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = rho * ux
                    host[b + IDX_MY] = rho * uy
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
                    host[b + IDX_BX] = Bx_v
                    host[b + IDX_BY] = By_v
                    host[b + IDX_BZ] = Real(0.0)
                    host[b + IDX_PSI] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)
    var mass_0 = _integrate_field[IDX_RHO](ctx, q, mesh)
    var e_0 = _integrate_field[IDX_E](ctx, q, mesh)

    print("time         ∫ρ dV            ∫E dV            ∫ψ² dV         rel Δmass   rel ΔE")
    print("-----------  ---------------- ---------------- ---------------- ----------  ----------")


    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MHD_GLM_NC],
        mut q_out: DGState2D[P, MHD_GLM_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_2d[P](ctx, q_in, q_out, C_H, C_P_SQ)

    @parameter
    def adjust(mut q_: DGState2D[P, MHD_GLM_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        # Snapshot current state, dump VTU, print diagnostics.
        ctx.synchronize()
        dump_imhd_glm_2d_frame[P, MHD_GLM_NC](fw.next_path(t), ctx, mesh, q, GAMMA)
        var mass = _integrate_field[IDX_RHO](ctx, q, mesh)
        var e_tot = _integrate_field[IDX_E](ctx, q, mesh)
        var psi2 = _integrate_squared_field[IDX_PSI](ctx, q, mesh)
        var dm = (mass - mass_0) / mass_0
        var dE = (e_tot - e_0) / e_0
        print(t, " ", mass, " ", e_tot, " ", psi2, " ", dm, " ", dE)

        # Advance to next frame time.
        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step[compute_rhs, adjust](ctx, q, dt_use)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
