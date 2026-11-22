# Convergence regression test: confirm that each polynomial order P attains the
# expected L2 rate of approximately (P + 1) on smooth advection.
#
# We test P = 1, 2, 3. P = 4 is excluded because at N ≥ 32 the L2 error already
# bottoms out at the SSPRK3 (third-order) time-integration floor and the
# observed convergence rate is dominated by that floor rather than the spatial
# discretization.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *
from warpxm.testing.convergence import check_convergence


def _wave(x: Real) -> Real:
    return sin(TWO_PI * x)


def _one_run[P: Int](
    var ctx: DeviceContext, num_elements: Int, dt: Real, t_final: Real,
) raises -> Real:
    comptime State = DGState[1, P, 1]
    var mesh = BlockMesh[1, P](ctx, Real(-0.5), Real(0.5), num_elements, periodic=True)
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, 1]:
        return [_wave(x)]
    mesh.fill(q, initial)

    var spatial = DGAdvectionSolver[1, P, 1](
        Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3[1, P, 1](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, no_op_adjust[1, P, 1],
    )
    ctx.synchronize()
    return mesh.l2_error(q, 0, _wave)


def _check_rate[P: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
) raises:
    """Check the observed rate and bound both errors absolutely.

    `tol_lo` / `tol_hi` are about 3x the measured coarse / fine errors, so
    a defect that scales the error uniformly (and leaves the rate intact)
    still fails.
    """
    var dt: Real = 5.0e-4
    var t_final: Real = 1.0
    var err_lo = _one_run[P](ctx, 32, dt, t_final)
    var err_hi = _one_run[P](ctx, 64, dt, t_final)
    _ = check_convergence(
        "P = " + String(P) + " (N = 32, 64)",
        err_lo, err_hi, expected, tol_lo, tol_hi,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0), Real(9.0e-2), Real(2.2e-2))
    _check_rate[2](ctx, Real(3.0), Real(3.0e-4), Real(4.0e-5))
    _check_rate[3](ctx, Real(4.0), Real(3.2e-6), Real(2.0e-7))
    print("PASSED")
