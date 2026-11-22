# 3D 5-moment Euler entropy-wave regression.
#
# IC (triply-periodic, smooth):
#     ρ(x, y, z, 0) = 1 + ε · sin(2π (x + y + z))
#     u = (1, 1, 1)   (uniform flow)
#     P = 1           (constant pressure)
#     E = P/(γ-1) + ½ρ|u|²
#
# This is a pure entropy / contact wave: pressure and velocity stay
# constant in space and time, density advects with the flow. After
# t = 1 (one period in each axis) the IC returns exactly.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime UX: Real = 1.0
comptime UY: Real = 1.0
comptime UZ: Real = 1.0
comptime State = DGState[3, P, EULER_NC]


def _rho_exact(x: Real, y: Real, z: Real) -> Real:
    return Real(1.0) + EPS * sin(TWO_PI * (x + y + z))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, EULER_NC]:
        var pp: Real = 1.0
        var rho = _rho_exact(x, y, z)
        var ke = Real(0.5) * rho * (UX * UX + UY * UY + UZ * UZ)
        return [rho, rho * UX, rho * UY, rho * UZ, pp / (GAMMA - Real(1.0)) + ke]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[3, P](app)
    var stepper = SSPRK3[3, P, EULER_NC](ctx, mesh.num_elements)

    # The wave moves along (1, 1, 1) at speed |u| = √3 with wavelength 1/√3,
    # so the solution returns to the IC every 1/3 time unit. One period is
    # enough to measure the dispersion / dissipation error; 334 steps of
    # dt = (1/3)/334 ≈ 1e-3 land exactly on it.
    var t_final: Real = Real(1.0) / Real(3.0)
    var dt: Real = t_final / Real(334)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    for _ in range(334):
        stepper.step(ctx, q, dt, compute_rhs, no_op_adjust[3, P, EULER_NC])
        t += dt
    ctx.synchronize()

    var l2 = mesh.l2_error(q, IDX_RHO, _rho_exact)
    var mass_drift: Real = 0.0
    def drift(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, EULER_NC],
    ) {mut mass_drift}:
        mass_drift += w * (v[IDX_RHO] - Real(1.0))
    mesh.visit_nodes(q, drift)
    print("3D Euler entropy wave — t =", t)
    print("  ρ L2 error vs exact:", l2)
    print("  net mass drift:      ", mass_drift)
    assert_true(l2 < Real(1.0e-4), "ρ L2 error too large")
    assert_true(fabs(mass_drift) < Real(1.0e-12), "mass not conserved")
    print("PASSED")
