# 1D Maxwell DG convergence regression — confirm rate ≈ (P + 1) on the
# smooth plane-wave problem. Extends the convergence-rate validation
# from scalar advection (`test_advection_convergence_1d`) to a multi-component
# vector physics — `MAXWELL_NC = 6` — which exercises the full DG
# kernel's per-component flux accumulation path.
#
# IC + analytic solution: a TM plane wave, `E_z = sin(2π x)`,
# `B_y = -sin(2π x) / c`, on a periodic unit interval. With `c = 1` the wave traverses one
# wavelength in `T = 1` and returns to the IC bit-for-bit modulo the
# DG truncation error; the L² error of `(E_z, B_y)` at `t = T` is what
# we fit.
#
# Sweep: `P ∈ {1, 2, 3}` on `NX ∈ {32, 64}`. CFL pinned to the finest
# mesh (`dt = 2.5·10⁻⁴`) so SSPRK3's `O(dt³)` doesn't pollute.
#
# Build / run:
#     pixi run mojo run -I src tests/test_maxwell_convergence_1d.mojo

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime C_LIGHT: Real = 1.0
comptime DT: Real = 2.5e-4


def _one_run[P: Int](
    var ctx: DeviceContext, num_x: Int,
) raises -> Real:
    comptime State = DGState[1, P, MAXWELL_NC]
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), num_x, periodic=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, MAXWELL_NC]:
        var s = sin(TWO_PI * x)
        return [Real(0.0), Real(0.0), s, Real(0.0), -s / C_LIGHT, Real(0.0)]
    mesh.fill(q, initial)

    var spatial = DGMaxwellSolver[1, P](
        Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)),
    )
    var stepper = SSPRK3[1, P, MAXWELL_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t_final: Real = 1.0 / C_LIGHT
    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, DT, compute_rhs,
        no_op_adjust[1, P, MAXWELL_NC],
    )
    ctx.synchronize()

    # Combined L² on the active (E_z, B_y) components.
    var acc: Real = 0.0
    var max_other: Real = 0.0
    def scan(x: Real, w: Real, v: Array[Real, MAXWELL_NC]) {mut acc, mut max_other}:
        var s = sin(TWO_PI * x)
        var d_Ez = v[IDX_EZ] - s
        var d_By = v[IDX_BY] - (-s / C_LIGHT)
        acc += w * (d_Ez * d_Ez + d_By * d_By)
        # E_x, E_y, B_x, B_z are not excited and must stay at 0.
        for idx in [IDX_EX, IDX_EY, IDX_BX, IDX_BZ]:
            max_other = fmax(max_other, fabs(v[idx]))
    mesh.visit_nodes(q, scan)
    assert_true(
        max_other < Real(1.0e-12),
        "unexcited components grew — 1D Maxwell coupling is spurious",
    )
    return sqrt(acc)


def _check_rate[P: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
) raises:
    """Check the observed rate between 32 and 64 elements and bound both
    errors absolutely (`tol_lo` / `tol_hi` are about 3x the measured
    errors)."""
    _ = check_convergence(
        "1D Maxwell P = " + String(P),
        _one_run[P](ctx, 32), _one_run[P](ctx, 64),
        expected, tol_lo, tol_hi,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0), Real(1.2e-1), Real(3.1e-2))
    _check_rate[2](ctx, Real(3.0), Real(4.5e-4), Real(5.5e-5))
    _check_rate[3](ctx, Real(4.0), Real(4.5e-6), Real(2.8e-7))
    print("PASSED")
