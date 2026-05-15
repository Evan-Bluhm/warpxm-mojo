# 2D Brio-Wu MHD shock tube with HLLD numerical flux + multi-frame VTU
# output. Showcases the full strong-MHD pipeline:
#   * `DGImhdSolver2D[P, USE_HLLD=True]` — Miyoshi-Kusano 5-state solver,
#     much sharper shocks than Rusanov.
#   * Moe-Rossmanith limiter for Gibbs control at the strong B_y jump.
#   * 1D physics extruded along a thin y-strip — analytic solution at
#     t = 0.1 has fast rarefactions, slow compound waves, contact, and
#     a slow shock, all visible in the dumped VTU frames.
#
# Per frame the driver prints (∫ρ dV, ∫B_x dV, max|Δρ/Δx|) so the sharpness
# of the rho-jump near the contact can be monitored.
#
# Build / run:
#     mkdir -p /tmp/briowu_hlld_2d
#     pixi run mojo run -I src examples/imhd_briowu_2d_hlld_vtu.mojo
#     # Open /tmp/briowu_hlld_2d/briowu.pvd in ParaView.

from std.math import isnan
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith_2d import MoeRossmanith2D
from warpxm.apps.imhd_2d import (
    IMHD2D, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_imhd_2d import dump_imhd_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 4
comptime GAMMA: Real = 2.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 1.0e-4
comptime NUM_FRAMES: Int = 11
comptime OUTPUT_DIR: String = "/tmp/briowu_hlld_2d"
comptime BASENAME: String = "briowu"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var rho: Real
                    var pp: Real
                    var By: Real
                    if x < Real(0.5):
                        rho = 1.0; pp = 1.0; By = 1.0
                    else:
                        rho = 0.125; pp = 0.1; By = -1.0
                    var Bx_v: Real = 0.75
                    var bsq = Bx_v * Bx_v + By * By
                    var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
                    host[b + IDX_BX] = Bx_v
                    host[b + IDX_BY] = By
                    host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var mass_0: Real = 0.0
    var bx_0: Real = 0.0
    for j in range(NY):
        for i in range(NX):
            var el = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var n = jn * P1 + in_
                    var b = (el * nodes_per_elem + n) * MHD_NC
                    var jw = ja * weights[in_] * weights[jn]
                    mass_0 += jw * host[b + IDX_RHO]
                    bx_0 += jw * host[b + IDX_BX]

    var app = IMHD2D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
    var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_NC](ctx, mesh.num_elements)
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)
    var mid_node = nodes_per_elem // 2

    print("frame  time   ∫ρ dV       ∫B_x dV     rel Δmass    rel ΔB_x     max|Δρ/Δx|")
    print("-----  -----  ----------  ----------  -----------  -----------  -----------")


    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState2D[P, MHD_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_imhd_2d_frame[P](fw.next_path(t), ctx, mesh, q, GAMMA)

        var mass: Real = 0.0
        var bx_int: Real = 0.0
        var max_grad: Real = 0.0
        var rho_row = InlineArray[Real, NX](fill=Real(0.0))
        with q.buf.map_to_host() as h:
            for j in range(NY):
                for i in range(NX):
                    var el = j * NX + i
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = jn * P1 + in_
                            var b = (el * nodes_per_elem + n) * MHD_NC
                            var jw = ja * weights[in_] * weights[jn]
                            mass += jw * h[b + IDX_RHO]
                            bx_int += jw * h[b + IDX_BX]
            # Sample j=0 row for sharpness.
            for i in range(NX):
                var el = 0 * NX + i
                var b = (el * nodes_per_elem + mid_node) * MHD_NC
                rho_row[i] = h[b + IDX_RHO]
            for i in range(NX - 1):
                var g = fabs(rho_row[i + 1] - rho_row[i])
                if g > max_grad: max_grad = g

        var dm = fabs(mass - mass_0) / mass_0
        var db = fabs(bx_int - bx_0) / bx_0
        print(frame, " ", t, " ", mass, " ", bx_int, " ", dm, " ", db, " ", max_grad)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step[compute_rhs, adjust](ctx, q, dt_use)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
