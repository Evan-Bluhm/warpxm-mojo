# Convergence study for 1D scalar advection.
#
# Advects sin(2π x) for one period on a periodic mesh and reports L2 error
# vs the exact (initial) solution for each (P, N) combination. With SSPRK3
# time stepping at small Δt the spatial error dominates, so the L2 error
# should scale like dx^(P + 1) for smooth solutions.
#
#     pixi run example convergence_advection_1d

from std.collections import Array
from std.math import log2, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *


def run_once[P: Int](
    var ctx: DeviceContext, num_elements: Int, dt: Real, t_final: Real,
) raises -> Real:
    """Run advection at order P with N elements and return the post-period L2 error."""
    comptime State = DGState[1, P, 1]
    var mesh = BlockMesh[1, P](ctx, Real(-0.5), Real(0.5), num_elements, periodic=True)
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def wave(x: Real) -> Real:
        return sin(TWO_PI * x)
    def initial(x: Real) -> Array[Real, 1]:
        return [sin(TWO_PI * x)]
    mesh.fill(q, initial)

    var app = Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    var spatial = DGAdvectionSolver[1, P, 1](app)
    var stepper = SSPRK3[1, P, 1](ctx, mesh.num_elements, mesh.num_ghost)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, no_op_adjust[1, P, 1],
    )
    return mesh.l2_error(q, 0, wave)


def sweep_p[P: Int](var ctx: DeviceContext) raises:
    """Run two resolutions at order P and print observed convergence rate."""
    # A convergence study wants the time error out of the way, so the step is
    # set by accuracy, not by the CFL limit (which would give dt ∝ dx and a
    # third-order time error that swamps the P = 4 spatial error). SSPRK3
    # with dt = 5e-4 over one period contributes ~ (5e-4)^3 ≈ 1e-10.
    var dt: Real = 5.0e-4
    var t_final: Real = 1.0

    var err_lo = run_once[P](ctx, 32, dt, t_final)
    var err_hi = run_once[P](ctx, 64, dt, t_final)
    var rate = log2(err_lo / err_hi)
    root_print(
        "  P =", P,
        " N = 32 ⇒ L2 =", err_lo,
        " | N = 64 ⇒ L2 =", err_hi,
        " | rate ≈", rate,
        " (expected ≈", P + 1, ")",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    root_print("1D advection convergence — sin(2π x) on [-0.5, 0.5], one period")
    sweep_p[1](ctx)
    sweep_p[2](ctx)
    sweep_p[3](ctx)
    sweep_p[4](ctx)
