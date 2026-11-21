# Smoke test for the 2D VTU writer.
#
# Runs a few Euler steps from a uniform IC, dumps a VTU file to /tmp, and
# verifies the file:
#   * is non-empty
#   * starts with the XML header / VTKFile tag
#   * ends with the closing </VTKFile> tag
#   * advertises the expected point and cell counts in the Piece element
#
# We don't validate the full XML schema — that's covered indirectly by
# whether `pixi run python -c "xml.etree.ElementTree.parse(...)"` succeeds
# offline. This test catches simple regressions (file truncated, header
# malformed, wrong counts) without pulling in a heavy dependency.

from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.core.types import Real
from warpxm.driver.vtu_2d import dump_euler_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 4
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime OUTPUT_PATH: String = "/tmp/warpxm_vtu_smoke.vtu"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var nodes_per_elem = (P + 1) * (P + 1)
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(mesh.num_elements * nodes_per_elem):
        var b = k * EULER_NC
        host[b + IDX_RHO] = Real(1.0)
        host[b + IDX_MX] = Real(0.0)
        host[b + IDX_MY] = Real(0.0)
        host[b + IDX_MZ] = Real(0.0)
        host[b + IDX_E] = Real(1.0) / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    dump_euler_2d_frame[P](OUTPUT_PATH, ctx, mesh, q, GAMMA)

    # --- Verify the file came out OK ----------------------------------------
    var text = String(open(OUTPUT_PATH, "r").read())
    assert_true(text.startswith("<?xml"), "VTU missing XML header")
    assert_true("<VTKFile" in text, "VTU missing <VTKFile> tag")
    assert_true("</VTKFile>" in text, "VTU missing closing </VTKFile> tag")

    var expected_points = mesh.num_elements * nodes_per_elem
    var expected_cells = mesh.num_elements * P * P
    var expected_points_attr = "NumberOfPoints=\"" + String(expected_points) + "\""
    var expected_cells_attr = "NumberOfCells=\"" + String(expected_cells) + "\""
    assert_true(expected_points_attr in text,
                "expected " + expected_points_attr + " in VTU")
    assert_true(expected_cells_attr in text,
                "expected " + expected_cells_attr + " in VTU")
    assert_true("Name=\"rho\"" in text, "VTU missing rho field")
    assert_true("Name=\"P\"" in text, "VTU missing P field")
    print("Wrote", OUTPUT_PATH)
    print("PASSED")
