# Dormand-Prince 5(4) stepper test on smooth 1D advection: the fixed-step
# path and the adaptive step-size controller.
#
# Problem: q_t + q_x = 0 on a periodic unit interval, q0 = sin(2π x), run
# for one period so the exact solution is the IC. P = 3, N = 64, where the
# spatial L2 error floor is about 6.7e-8.
#
# Checks:
#   (a) Fixed-step `step` reaches the spatial floor at a small dt.
#   (b) Every adaptive step obeys the controller contract: the step taken
#       is never longer than the one requested, and the suggested next
#       step is within the [0.2, 5] clip of the step taken.
#   (c) Rejection path: a first step far beyond the stability limit is
#       shrunk, and the run still ends accurate.
#   (d) The controller responds to the tolerance: tightening it by 100x
#       takes more (but not wildly more) steps and reduces the error, and
#       the loose-tolerance error is time-integration dominated, i.e. the
#       tolerance really is what sets it.
#   (e) Step counts sit in a band around the measured values, so both an
#       over-cautious and an over-eager controller fail.

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 3
comptime NX: Int = 64
comptime T_FINAL: Real = 1.0


@fieldwise_init
struct RunResult(Copyable):
    var l2_err: Real
    var num_steps: Int
    var min_dt: Real
    var max_dt: Real
    var first_dt_used: Real


def _run(
    var ctx: DeviceContext, num_x: Int,
    adaptive: Bool, dt_init: Real, atol: Real, rtol: Real,
) raises -> RunResult:
    """One period. `adaptive=False` takes fixed steps of `dt_init`."""
    var mesh = BlockMesh1D[P](ctx, Real(-0.5), Real(0.5), num_x, periodic=True)
    var q = DGState1D[P, 1](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.total_nodes)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            host[e * nodes_per_elem + i] = sin(TWO_PI * x)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGSpatialSolver1D[P, 1](
        Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = DormandPrince45_1D[P, 1](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState1D[P, 1],
        mut q_out: DGState1D[P, 1],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = dt_init
    var num_steps: Int = 0
    var min_dt: Real = 1.0e30
    var max_dt: Real = 0.0
    var first_dt_used: Real = 0.0
    while t < T_FINAL - Real(1.0e-12):
        var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
        var dt_used = dt_try
        if adaptive:
            var ret = stepper.step_adaptive(
                ctx, q, dt_try, compute_rhs, atol=atol, rtol=rtol,
            )
            dt_used = ret[0]
            var dt_next = ret[1]
            # Controller contract.
            assert_true(dt_used > Real(0.0), "non-positive step taken")
            assert_true(
                dt_used <= dt_try * (Real(1.0) + Real(1.0e-12)),
                "step taken is longer than the step requested",
            )
            assert_true(
                dt_next >= Real(0.2) * dt_used * (Real(1.0) - Real(1.0e-12))
                and dt_next <= Real(5.0) * dt_used * (Real(1.0) + Real(1.0e-12)),
                "suggested next step outside the [0.2, 5] clip",
            )
            dt = dt_next
        else:
            stepper.step(ctx, q, dt_try, compute_rhs)
        if num_steps == 0: first_dt_used = dt_used
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        assert_true(num_steps < 100000, "step count ran away")
    ctx.synchronize()

    var l2_err: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var d = h[e * nodes_per_elem + i] - sin(TWO_PI * x)
                l2_err += weights[i] * mesh.jacobian * d * d
    return RunResult(
        l2_err ** Real(0.5), num_steps, min_dt, max_dt, first_dt_used,
    )


def _report(label: String, r: RunResult):
    print(
        label, " steps =", r.num_steps, " dt in [", r.min_dt, ",", r.max_dt,
        "]  first dt =", r.first_dt_used, " L2 =", r.l2_err,
    )


def _in_band(n: Int, measured: Int) -> Bool:
    """Step count within 15% of the measured value."""
    return (
        Real(n) > Real(0.85) * Real(measured)
        and Real(n) < Real(1.15) * Real(measured)
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # (a) Fixed step, dt well inside stability: spatial floor (6.7e-8).
    var fixed = _run(ctx, NX, False, Real(5.0e-4), Real(0.0), Real(0.0))
    _report("fixed dt=5e-4        ", fixed)
    assert_true(fixed.num_steps == 2000, "fixed-step run took the wrong number of steps")
    assert_true(fixed.l2_err < Real(2.0e-7), "fixed-step DP45 error above the spatial floor")

    # (d) N = 64 is stability-limited (dt ~ 5.5e-3 either way), so the
    # tolerance shows up in the error, not the step count.
    # Measured: 183 steps / 2.08e-4 and 186 steps / 3.14e-7.
    var loose = _run(ctx, NX, True, Real(5.0e-3), Real(1.0e-5), Real(1.0e-3))
    _report("N=64 rtol=1e-3       ", loose)
    var tight = _run(ctx, NX, True, Real(5.0e-3), Real(1.0e-7), Real(1.0e-5))
    _report("N=64 rtol=1e-5       ", tight)
    assert_true(_in_band(loose.num_steps, 183), "rtol=1e-3 step count outside band")
    assert_true(_in_band(tight.num_steps, 186), "rtol=1e-5 step count outside band")
    assert_true(loose.l2_err < Real(6.5e-4), "rtol=1e-3 error too large")
    assert_true(tight.l2_err < Real(1.0e-6), "rtol=1e-5 error too large")
    assert_true(
        loose.l2_err > Real(100.0) * tight.l2_err,
        "100x tighter tolerance did not reduce the error by 100x",
    )
    assert_true(
        loose.l2_err > Real(100.0) * fixed.l2_err,
        "loose-tolerance error is not time-integration dominated",
    )

    # (c) Rejection path: dt0 = 0.2 is ~40x the stability limit. Measured:
    # two rejections (first step 0.2 * 0.2^2 = 8e-3), 186 steps, L2 4.7e-7.
    var reject = _run(ctx, NX, True, Real(0.2), Real(1.0e-7), Real(1.0e-5))
    _report("N=64 rtol=1e-5 dt0=.2", reject)
    assert_true(reject.first_dt_used < Real(0.05), "oversized first step was not rejected")
    assert_true(reject.first_dt_used > Real(1.0e-3), "first step shrunk far more than needed")
    assert_true(_in_band(reject.num_steps, 186), "dt0=0.2 step count outside band")
    assert_true(reject.l2_err < Real(1.5e-6), "run with rejected first step is inaccurate")

    # (d) N = 16 at tight tolerances is accuracy-limited, so the step count
    # must follow the 5th-order law: 100x tighter => 100^(1/5) = 2.51x more
    # steps. Measured: 91 and 237 steps (ratio 2.60); L2 is the N = 16
    # spatial floor, 1.70e-5, in both.
    var acc8 = _run(ctx, 16, True, Real(5.0e-3), Real(1.0e-10), Real(1.0e-8))
    _report("N=16 rtol=1e-8       ", acc8)
    var acc10 = _run(ctx, 16, True, Real(5.0e-3), Real(1.0e-12), Real(1.0e-10))
    _report("N=16 rtol=1e-10      ", acc10)
    var ratio = Real(acc10.num_steps) / Real(acc8.num_steps)
    print("N=16 step ratio for 100x tighter tolerance:", ratio, "(theory 2.51)")
    assert_true(_in_band(acc8.num_steps, 91), "N=16 rtol=1e-8 step count outside band")
    assert_true(_in_band(acc10.num_steps, 237), "N=16 rtol=1e-10 step count outside band")
    assert_true(
        ratio > Real(2.2) and ratio < Real(3.0),
        "step count does not scale as tol^(-1/5)",
    )
    assert_true(acc8.l2_err < Real(5.0e-5), "N=16 rtol=1e-8 error too large")
    assert_true(acc10.l2_err < Real(5.0e-5), "N=16 rtol=1e-10 error too large")
    # rtol = 1e-10 cannot accept the 5e-3 first step.
    assert_true(acc10.first_dt_used < Real(5.0e-3), "tight tolerance accepted the first step")
    print("PASSED")
