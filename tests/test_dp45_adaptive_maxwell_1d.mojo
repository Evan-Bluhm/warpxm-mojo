# Adaptive DP45 on 1D Maxwell — completes DP45 + Maxwell coverage
# across all three dimensions. Same plane-wave IC as
# test_maxwell_1d_plane_wave.mojo.

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NX: Int = 32
comptime C_LIGHT: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](ctx, Real(0.0), Real(1.0), NX, periodic=True)
    var q = DGState1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var s = sin(TWO_PI * x)
            var b = (e * nodes_per_elem + i) * MAXWELL_NC
            host[b + IDX_EX] = Real(0.0)
            host[b + IDX_EY] = Real(0.0)
            host[b + IDX_EZ] = s
            host[b + IDX_BX] = Real(0.0)
            host[b + IDX_BY] = -s / C_LIGHT
            host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGMaxwellSolver1D[P](Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)))
    var stepper = DormandPrince45_1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, MAXWELL_NC],
        mut q_out: DGState1D[P, MAXWELL_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = DT_INIT
    var num_steps: Int = 0
    var min_dt = dt
    var max_dt: Real = 0.0
    while t < T_FINAL - Real(1.0e-12):
        var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
        var ret = stepper.step_adaptive[compute_rhs](
            ctx, q, dt_try, atol=ATOL, rtol=RTOL,
        )
        var dt_used = ret[0]
        var dt_next = ret[1]
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break
    ctx.synchronize()

    var err_Ez: Real = 0.0
    var err_By: Real = 0.0
    var max_other: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var s = sin(TWO_PI * x)
                var b = (e * nodes_per_elem + i) * MAXWELL_NC
                var d_Ez = h[b + IDX_EZ] - s
                var d_By = h[b + IDX_BY] - (-s / C_LIGHT)
                var jw = mesh.jacobian * weights[i]
                err_Ez += jw * d_Ez * d_Ez
                err_By += jw * d_By * d_By
                for idx in [IDX_EX, IDX_EY, IDX_BX, IDX_BZ]:
                    var v = h[b + idx]
                    if fabs(v) > max_other: max_other = fabs(v)
    var l2_Ez = err_Ez ** Real(0.5)
    var l2_By = err_By ** Real(0.5)

    print("Adaptive DP45 1D Maxwell (P=2, NX=32, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 E_z          :", l2_Ez)
    print("  L2 B_y          :", l2_By)
    print("  max |E_x,E_y,B_x,B_z| :", max_other)

    # Fixed-dt SSPRK3 baseline is ~2000 steps at dt=5e-4.
    assert_true(num_steps < 300, "Adaptive DP45 took too many steps")
    assert_true(l2_Ez < Real(1.0e-3), "E_z error worse than baseline")
    assert_true(l2_By < Real(1.0e-3), "B_y error worse than baseline")
    assert_true(max_other < Real(1.0e-7),
                "Transverse components grew — coupling spurious")
    assert_true(max_dt > min_dt * Real(1.5), "Controller did not vary dt")
    print("PASSED")
