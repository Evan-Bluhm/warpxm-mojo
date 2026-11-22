# 2D Maxwell plane-wave regression test.
#
# IC: a TM-mode wave with E along ẑ, propagating in (1,1)/√2 direction at
# wavelength λ = 1/√2 (one wavelength on the diagonal of the unit cell).
#
#     E_z(x, y, 0) = sin( 2π (x + y) )
#     B_x(x, y, 0) =  (1/√2) sin( 2π (x + y) )
#     B_y(x, y, 0) = -(1/√2) sin( 2π (x + y) )
#     E_x = E_y = B_z = 0
#
# With c = 1, the period is T = λ/c = 1/√2. After exactly one period the
# solution must return to the IC. We check L2 error against the IC.
#
# This exercises every component of the Maxwell flux tensor (the wave
# couples E_z ↔ B_x ↔ B_y in the curl pattern) and validates that the DG
# kernel's volume + face flux logic handles the linear hyperbolic system
# correctly.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime C_LIGHT: Real = 1.0
comptime State = DGState[2, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    var inv_sqrt2: Real = Real(1.0) / sqrt(Real(2.0))

    def initial(x: Real, y: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var s = sin(TWO_PI * (x + y))
        return [Real(0.0), Real(0.0), s, inv_sqrt2 * s, -inv_sqrt2 * s, Real(0.0)]
    mesh.fill(q, initial)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver[2, P](app)
    var stepper = SSPRK3[2, P, MAXWELL_NC](ctx, mesh.num_elements)

    # Period: T = λ / c, with λ = 1/√2 → T = 1/√2 ≈ 0.7071.
    var t_final: Real = inv_sqrt2
    var dt: Real = Real(5.0e-4)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    # T / dt is not an integer; the run stops at the last whole step
    # before T (within half a step of it).
    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_op_adjust[2, P, MAXWELL_NC])
        t += dt_use
    ctx.synchronize()

    # L2 errors per component, against the IC.
    def exact_ez(x: Real, y: Real) -> Real:
        return sin(TWO_PI * (x + y))
    def exact_bx(x: Real, y: Real) {imm} -> Real:
        return inv_sqrt2 * sin(TWO_PI * (x + y))
    def exact_by(x: Real, y: Real) {imm} -> Real:
        return -inv_sqrt2 * sin(TWO_PI * (x + y))
    var l2_ez = mesh.l2_error(q, IDX_EZ, exact_ez)
    var l2_bx = mesh.l2_error(q, IDX_BX, exact_bx)
    var l2_by = mesh.l2_error(q, IDX_BY, exact_by)
    print("L2 E_z =", l2_ez)
    print("L2 B_x =", l2_bx)
    print("L2 B_y =", l2_by)
    assert_true(l2_ez < Real(5.0e-3), "E_z error too large")
    assert_true(l2_bx < Real(5.0e-3), "B_x error too large")
    assert_true(l2_by < Real(5.0e-3), "B_y error too large")
    print("PASSED")
