# 1D 5-moment Euler entropy-wave regression.
#
# A smooth density perturbation `ρ(x, 0) = 1 + ε sin(2π x)` riding on a
# uniform flow `u = 1`, with constant pressure `P = 1`. Pure entropy waves
# are linearly degenerate: density advects at speed u while pressure and
# velocity stay constant. After one period the solution returns to the IC.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 64
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime U: Real = 1.0
comptime State = DGState[1, P, EULER_NC]


def _rho_exact(x: Real) -> Real:
    return Real(1.0) + EPS * sin(TWO_PI * x)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, EULER_NC]:
        var pp: Real = 1.0
        var rho = _rho_exact(x)
        var ke = Real(0.5) * rho * U * U
        return [rho, rho * U, Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0)) + ke]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[1, P](app)
    var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs,
        no_op_adjust[1, P, EULER_NC],
    )
    ctx.synchronize()

    var l2 = mesh.l2_error(q, IDX_RHO, _rho_exact)
    var mass_drift: Real = 0.0
    def drift(x: Real, w: Real, v: Array[Real, EULER_NC]) {mut mass_drift}:
        mass_drift += w * (v[IDX_RHO] - Real(1.0))
    mesh.visit_nodes(q, drift)
    print("ρ L2 error vs exact:", l2)
    print("net mass drift:", mass_drift)
    assert_true(l2 < Real(1.0e-4), "Euler 1D entropy-wave error too large")
    assert_true(mass_drift < Real(1.0e-12) and mass_drift > -Real(1.0e-12),
                "mass not conserved")
    print("PASSED")
