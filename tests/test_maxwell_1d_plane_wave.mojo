# 1D Maxwell plane-wave regression — the lowest-rank case of the
# Maxwell solver. Validates that the consolidated app + DG core
# architecture handles 1D Maxwell cleanly (no per-dim app rewrite
# needed; the canonical Maxwell struct takes a 3-component normal and
# the 1D DG kernel passes (±1, 0, 0)).
#
# IC: a TM wave propagating along x̂ with E_z and B_y:
#     E_z(x, 0) =  sin(2π x)
#     B_y(x, 0) = -sin(2π x) / c
#     all others = 0
#
# With c = 1 the wave traverses one wavelength in T = 1 and returns to
# the IC. After one period, L2 errors on E_z / B_y should be small and
# all other components must stay at 0.

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
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
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NX: Int = 32
comptime C_LIGHT: Real = 1.0
comptime DT: Real = 5.0e-4


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
    var stepper = SSPRK3_1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState1D[P, MAXWELL_NC],
        mut q_out: DGState1D[P, MAXWELL_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState1D[P, MAXWELL_NC]) raises {imm}:
        pass

    var t_final: Real = 1.0 / C_LIGHT
    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
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
                # E_x, E_y, B_x, B_z must stay at 0.
                for idx in [IDX_EX, IDX_EY, IDX_BX, IDX_BZ]:
                    var v = h[b + idx]
                    if fabs(v) > max_other: max_other = fabs(v)
    var l2_Ez = err_Ez ** Real(0.5)
    var l2_By = err_By ** Real(0.5)
    print("1D Maxwell plane wave — t =", t)
    print("  L2 E_z error:", l2_Ez)
    print("  L2 B_y error:", l2_By)
    print("  max |E_x, E_y, B_x, B_z| (should be ~0):", max_other)
    assert_true(l2_Ez < Real(1.0e-3), "E_z error too large")
    assert_true(l2_By < Real(1.0e-3), "B_y error too large")
    assert_true(max_other < Real(1.0e-12),
                "Transverse components grew — 1D Maxwell coupling is spurious")
    print("PASSED")
