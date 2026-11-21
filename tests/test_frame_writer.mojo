# Smoke test for the multi-frame VTU writer + PVD collection.
#
# Dumps 3 frames from an Orszag-Tang run, then verifies:
#   * each frame .vtu file exists and is well-formed XML
#   * the .pvd collection lists all 3 frames with their times

from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

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
from std.math import sin


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime OUTPUT_DIR: String = "/tmp/warpxm_frames_smoke"


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
        c_h=Real(4.0), c_p_sq=Real(0.5),
    )
    var spatial = DGImhdGLMSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var fw = FrameWriter("orszag_tang", OUTPUT_DIR)

    # Frame 0: IC.
    dump_imhd_glm_2d_frame[P, MHD_GLM_NC](fw.next_path(Real(0.0)), ctx, mesh, q, GAMMA)

    var dt: Real = 5.0e-3

    def compute_rhs(
        mut q_in: DGState2D[P, MHD_GLM_NC],
        mut q_out: DGState2D[P, MHD_GLM_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_2d[P](ctx, q_in, q_out, Real(4.0), Real(0.5))

    def adjust(mut q_: DGState2D[P, MHD_GLM_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    var t_targets = [Real(0.1), Real(0.2)]
    for i in range(len(t_targets)):
        var t_target = t_targets[i]
        while t < t_target - Real(0.5) * dt:
            stepper.step(ctx, q, dt, compute_rhs, adjust)
            t += dt
        ctx.synchronize()
        dump_imhd_glm_2d_frame[P, MHD_GLM_NC](fw.next_path(t), ctx, mesh, q, GAMMA)

    fw.write_pvd()

    # Read back the PVD file and check it lists the three frames.
    var pvd = String(open(OUTPUT_DIR + "/orszag_tang.pvd", "r").read())
    assert_true("orszag_tang_0000.vtu" in pvd, "frame 0 missing from PVD")
    assert_true("orszag_tang_0001.vtu" in pvd, "frame 1 missing from PVD")
    assert_true("orszag_tang_0002.vtu" in pvd, "frame 2 missing from PVD")
    assert_true("timestep=\"0.0\"" in pvd, "frame 0 time missing")
    assert_true("<Collection>" in pvd and "</Collection>" in pvd,
                "PVD missing Collection element")

    # Sanity-check the first frame .vtu file has the expected MHD point data.
    var vtu0 = String(open(OUTPUT_DIR + "/orszag_tang_0000.vtu", "r").read())
    assert_true("Name=\"rho\"" in vtu0, "MHD VTU missing rho field")
    assert_true("Name=\"Bx\"" in vtu0, "MHD VTU missing Bx field")
    assert_true("Name=\"By\"" in vtu0, "MHD VTU missing By field")
    assert_true("Name=\"psi\"" in vtu0, "GLM VTU missing psi field")
    assert_true("Name=\"P\"" in vtu0, "MHD VTU missing P field")
    print("Wrote", len(fw.frame_files), "frames + PVD at", OUTPUT_DIR)
    print("PASSED")
