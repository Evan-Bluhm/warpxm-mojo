# Sanity test for the 1D Euler DG solver.
#
# Initial condition: constant uniform state with zero velocity. Volume flux
# `f_x = (0, P, 0, 0, 0)` is constant in space → zero volume contribution.
# Numerical-flux jumps are zero across every face. RHS must be exactly zero.

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime State = DGState[1, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=True,
    )
    var q = State(ctx, mesh.num_elements)

    var rho: Real = 1.0
    var p: Real = 1.0
    var energy: Real = p / (GAMMA - Real(1.0))

    def initial(x: Real) {imm} -> Array[Real, EULER_NC]:
        return [rho, Real(0.0), Real(0.0), Real(0.0), energy]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[1, P](app)
    var rhs = State(ctx, mesh.num_elements)

    spatial.compute_rhs(ctx, mesh, q, rhs)
    ctx.synchronize()

    var max_abs: Real = 0.0
    def largest(x: Real, w: Real, v: Array[Real, EULER_NC]) {mut max_abs}:
        for c in range(EULER_NC):
            var a = fabs(v[c])
            if a > max_abs: max_abs = a
    mesh.visit_nodes(rhs, largest)
    print("max |rhs| on stationary state:", max_abs)
    assert_true(max_abs < Real(1.0e-12), "stationary 1D Euler RHS not zero")

    # Also evolve a few steps and confirm drift is exactly zero.
    var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var dt: Real = 1.0e-3
    for _ in range(50):
        stepper.step(ctx, q, dt, compute_rhs, no_op_adjust[1, P, EULER_NC])
    ctx.synchronize()

    var max_dev: Real = 0.0
    def deviation(x: Real, w: Real, v: Array[Real, EULER_NC]) {mut max_dev, imm}:
        var d_rho = fabs(v[IDX_RHO] - rho)
        var d_e = fabs(v[IDX_E] - energy)
        if d_rho > max_dev: max_dev = d_rho
        if d_e > max_dev: max_dev = d_e
    mesh.visit_nodes(q, deviation)
    print("max deviation after 50 steps:", max_dev)
    assert_true(max_dev < Real(1.0e-12), "stationary 1D Euler drifted")
    print("PASSED")
