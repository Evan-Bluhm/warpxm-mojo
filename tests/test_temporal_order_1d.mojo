# Temporal order of accuracy of the fixed-step integrators.
#
# Every other accuracy test in the suite is dominated by the spatial error,
# so a wrong Butcher coefficient that kept the method consistent would go
# unnoticed. This test isolates the time integrator by the method of
# lines: the DG semi-discretisation on a fixed mesh is an ODE system
# `dq/dt = L(q)`, and its exact solution is approximated by a reference
# run with a very small step. The error of a run with step `dt` against
# that reference contains no spatial error at all, so halving `dt` must
# reduce it by `2^order`:
#
#   * `SSPRK3.step`            — order 3 (ratio 8)
#   * `DormandPrince45.step`   — order 5 (ratio 32)
#
# The problem is a smooth, genuinely nonlinear 1D Euler flow (density,
# velocity and pressure all perturbed), so the nonlinear order conditions
# are exercised, not just the stability polynomial. It is stopped long
# before a shock can form.
#
# Build / run:
#     pixi run mojo run -I src tests/test_temporal_order_1d.mojo

from std.collections import Array
from std.math import cos, log2, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime T_FINAL: Real = 0.2
comptime N_COARSE: Int = 32      # dt = 6.25e-3
comptime N_FINE: Int = 64        # dt = 3.125e-3
comptime N_REF: Int = 1024       # DP45 reference, dt = 1.95e-4
comptime State = DGState[1, P, EULER_NC]


def _run[USE_DP45: Bool](var ctx: DeviceContext, n_steps: Int) raises -> List[Real]:
    """Integrate to `T_FINAL` in `n_steps` equal steps; return the state."""
    var mesh = BlockMesh[1, P](ctx, Real(0.0), Real(1.0), NX, periodic=True)
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, EULER_NC]:
        var rho = Real(1.0) + Real(0.2) * sin(TWO_PI * x)
        var u = Real(0.3) + Real(0.2) * cos(TWO_PI * x)
        var pp = Real(1.0) + Real(0.2) * cos(TWO_PI * x + Real(1.0))
        return [
            rho, rho * u, Real(0.0), Real(0.0),
            pp / (GAMMA - Real(1.0)) + Real(0.5) * rho * u * u,
        ]
    mesh.fill(q, initial)

    var spatial = DGEulerSolver[1, P](
        Euler(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var dt = T_FINAL / Real(n_steps)
    comptime if USE_DP45:
        var stepper = DormandPrince45[1, P, EULER_NC](ctx, mesh.num_elements)
        for _ in range(n_steps):
            stepper.step(ctx, q, dt, compute_rhs)
    else:
        var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)
        for _ in range(n_steps):
            stepper.step(ctx, q, dt, compute_rhs, no_op_adjust[1, P, EULER_NC])
    ctx.synchronize()

    var out = List[Real]()
    with q.buf.map_to_host() as h:
        for i in range(q.buf_len):
            out.append(h[i])
    return out^


def _rms_diff(a: List[Real], b: List[Real]) -> Real:
    var acc: Real = 0.0
    for i in range(len(a)):
        var d = a[i] - b[i]
        acc += d * d
    return sqrt(acc / Real(len(a)))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var q_ref = _run[True](ctx, N_REF)

    var rk3_coarse = _rms_diff(_run[False](ctx, N_COARSE), q_ref)
    var rk3_fine = _rms_diff(_run[False](ctx, N_FINE), q_ref)
    var rk3_order = log2(rk3_coarse / rk3_fine)
    print("SSPRK3: err(dt) =", rk3_coarse, " err(dt/2) =", rk3_fine, " order =", rk3_order)

    var dp_coarse = _rms_diff(_run[True](ctx, N_COARSE), q_ref)
    var dp_fine = _rms_diff(_run[True](ctx, N_FINE), q_ref)
    var dp_order = log2(dp_coarse / dp_fine)
    print("DP45  : err(dt) =", dp_coarse, " err(dt/2) =", dp_fine, " order =", dp_order)

    # Measured orders 2.992 and 5.002; errors 1.29e-5 and 3.03e-9.
    assert_true(rk3_order > Real(2.85) and rk3_order < Real(3.15), "SSPRK3 is not third order")
    assert_true(dp_order > Real(4.85) and dp_order < Real(5.15), "DP45 is not fifth order")
    assert_true(rk3_coarse < Real(4.0e-5), "SSPRK3 temporal error above tolerance")
    assert_true(dp_coarse < Real(9.0e-9), "DP45 temporal error above tolerance")
    print("PASSED")
