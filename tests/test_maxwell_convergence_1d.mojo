# 1D Maxwell DG convergence regression — confirm rate ≈ (P + 1) on the
# smooth plane-wave problem. Extends the convergence-rate validation
# from scalar advection (`test_dg_convergence_1d`) to a multi-component
# vector physics — `MAXWELL_NC = 6` — which exercises the full DG
# kernel's per-component flux accumulation path.
#
# IC + analytic solution: same TM-wave setup as
# `test_maxwell_1d_plane_wave`. With `c = 1` the wave traverses one
# wavelength in `T = 1` and returns to the IC bit-for-bit modulo the
# DG truncation error; the L² error of `(E_z, B_y)` at `t = T` is what
# we fit.
#
# Sweep: `P ∈ {1, 2, 3}` on `NX ∈ {32, 64}`. CFL pinned to the finest
# mesh (`dt = 2.5·10⁻⁴`) so SSPRK3's `O(dt³)` doesn't pollute.
#
# Build / run:
#     pixi run mojo run -I src tests/test_maxwell_convergence_1d.mojo

from std.math import log2, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EZ, IDX_BY,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime C_LIGHT: Real = 1.0
comptime DT: Real = 2.5e-4


def _one_run[P: Int](
    var ctx: DeviceContext, num_x: Int,
) raises -> Real:
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), num_x, periodic=True,
    )
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
            comptime for c in range(MAXWELL_NC):
                host[b + c] = Real(0.0)
            host[b + IDX_EZ] = s
            host[b + IDX_BY] = -s / C_LIGHT
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGMaxwellSolver1D[P](
        Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)),
    )
    var stepper = SSPRK3_1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, MAXWELL_NC],
        mut q_out: DGState1D[P, MAXWELL_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState1D[P, MAXWELL_NC]) raises:
        pass

    var t_final: Real = 1.0 / C_LIGHT
    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    # Combined L² on the active (E_z, B_y) components.
    var acc: Real = 0.0
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
                acc += jw * (d_Ez * d_Ez + d_By * d_By)
    return acc ** Real(0.5)


def _check_rate[P: Int](var ctx: DeviceContext, expected: Real) raises:
    var err_lo = _one_run[P](ctx, 32)
    var err_hi = _one_run[P](ctx, 64)
    var rate = log2(err_lo / err_hi)
    print(
        "P =", P,
        " err_NX32 =", err_lo,
        " err_NX64 =", err_hi,
        " rate =", rate,
    )
    assert_true(
        rate > expected - Real(0.25),
        "1D Maxwell convergence rate below tolerance",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0))
    _check_rate[2](ctx, Real(3.0))
    _check_rate[3](ctx, Real(4.0))
    print("PASSED")
