# 1D 5-moment Euler DG convergence regression — confirm rate ≈ (P + 1)
# on the smooth entropy-wave problem. Extends the convergence-rate
# validation from linear scalar advection (`test_advection_convergence_1d`)
# and linear vector electromagnetic (`test_maxwell_convergence_1d`) to
# a **nonlinear** physics: Euler's momentum-flux is quadratic in `m_x`
# (via `m_x² / ρ`), so this test exercises a code path that the linear
# tests don't.
#
# IC + analytic solution:
#     ρ(x, 0) = 1 + ε sin(2π x)
#     u       = 1   (uniform)
#     P       = 1   (uniform)
# Pure entropy waves are linearly degenerate — `u` and `P` stay
# constant, only `ρ` advects at speed `u`. Exact solution at `t = T`:
#     ρ(x, T) = 1 + ε sin(2π (x − u T))
#
# Sweep: `P ∈ {1, 2, 3}` on `NX ∈ {32, 64}`. `T = 1` (one full period).
# `dt = 5·10⁻⁴` pinned to the finest-mesh CFL so SSPRK3's `O(dt³)`
# doesn't pollute. Both errors are also bounded absolutely (about 3x the
# measured values) so a defect that leaves the rate intact still fails.
#
# Build / run:
#     pixi run mojo run -I src tests/test_euler_convergence_1d.mojo

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *
from warpxm.testing.convergence import check_convergence


comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime U: Real = 1.0
comptime DT: Real = 5.0e-4
comptime T_FINAL: Real = 1.0


def _rho_exact(x: Real) -> Real:
    """Density at t = 0 and, since u·T = 1, after one period."""
    return Real(1.0) + EPS * sin(TWO_PI * x)


def _one_run[P: Int](
    var ctx: DeviceContext, num_x: Int,
) raises -> Real:
    comptime State = DGState[1, P, EULER_NC]
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), num_x, periodic=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, EULER_NC]:
        var pp: Real = 1.0
        var rho = _rho_exact(x)
        var ke = Real(0.5) * rho * U * U
        return [rho, rho * U, Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0)) + ke]
    mesh.fill(q, initial)

    var spatial = DGEulerSolver[1, P](
        Euler(
            gas_gamma=GAMMA,
            min_density=Real(1e-12),
            min_pressure=Real(1e-12),
            cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs,
        no_op_adjust[1, P, EULER_NC],
    )
    ctx.synchronize()

    return mesh.l2_error(q, IDX_RHO, _rho_exact)


def _check_rate[P: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
) raises:
    var err_lo = _one_run[P](ctx, 32)
    var err_hi = _one_run[P](ctx, 64)
    _ = check_convergence(
        "1D Euler entropy wave, P = " + String(P),
        err_lo, err_hi, expected, tol_lo, tol_hi,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0), Real(8.6e-4), Real(2.2e-4))
    _check_rate[2](ctx, Real(3.0), Real(5.5e-6), Real(7.2e-7))
    _check_rate[3](ctx, Real(4.0), Real(2.3e-8), Real(1.5e-9))
    print("PASSED")
