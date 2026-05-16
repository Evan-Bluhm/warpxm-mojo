# Smoke test for the 3D VTU writer.
#
# Runs a quick 3D Euler blast wave (a few timesteps) and dumps a VTU.
# Verifies:
#   * file is well-formed XML
#   * Piece advertises the expected NumberOfPoints / NumberOfCells
#   * rho / u / v / w / P fields are present

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.euler import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler import Euler
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.driver.vtu_3d import dump_euler_3d_frame
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 4
comptime NY: Int = 4
comptime NZ: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime OUTPUT_PATH: String = "/tmp/warpxm_blast_3d.vtu"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        var zl = mesh.z_min + Real(k) * mesh.dz
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var r2 = (x - Real(0.5)) * (x - Real(0.5))
                            r2 += (y - Real(0.5)) * (y - Real(0.5))
                            r2 += (z - Real(0.5)) * (z - Real(0.5))
                            var pp = Real(1.0) + Real(3.0) * exp(-r2 / Real(0.04))
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = Real(1.0)
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, EULER_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState3D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    var dt: Real = 5.0e-3
    for _ in range(20):
        stepper.step[compute_rhs, adjust](ctx, q, dt)
    ctx.synchronize()

    dump_euler_3d_frame[P](OUTPUT_PATH, ctx, mesh, q, GAMMA)

    var text = String(open(OUTPUT_PATH, "r").read())
    assert_true(text.startswith("<?xml"), "3D VTU missing XML header")
    assert_true("</VTKFile>" in text, "3D VTU missing closing tag")

    var expected_points = mesh.num_elements * nodes_per_elem
    var expected_cells = mesh.num_elements * P * P * P
    var expected_points_attr = "NumberOfPoints=\"" + String(expected_points) + "\""
    var expected_cells_attr = "NumberOfCells=\"" + String(expected_cells) + "\""
    assert_true(expected_points_attr in text,
                "expected " + expected_points_attr + " in 3D VTU")
    assert_true(expected_cells_attr in text,
                "expected " + expected_cells_attr + " in 3D VTU")
    assert_true("Name=\"rho\"" in text, "3D VTU missing rho")
    assert_true("Name=\"u\"" in text, "3D VTU missing u")
    assert_true("Name=\"v\"" in text, "3D VTU missing v")
    assert_true("Name=\"w\"" in text, "3D VTU missing w")
    assert_true("Name=\"P\"" in text, "3D VTU missing P")
    print("Wrote", OUTPUT_PATH)
    print("PASSED")
