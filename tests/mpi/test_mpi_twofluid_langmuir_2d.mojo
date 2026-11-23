# Multi-rank regression: electron fluid + Maxwell, coupled (Langmuir
# oscillation, as `tests/test_langmuir_2d.mojo`).
#
# Two fields of different widths (Euler, 5 components; Maxwell, 6) share
# one mesh and one halo plan, and the coupled stepper advances both. One
# plasma period returns the density perturbation to its initial phase.

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.plasma import *

comptime P: Int = 2
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = -1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime FluidState = DGState[2, P, EULER_NC]
comptime MaxwellState = DGState[2, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.5),
        nx=20, ny=6, periodic_x=True, periodic_y=True,
    )
    var q_fluid = FluidState(ctx, mesh.num_elements, mesh.num_ghost)
    var q_maxwell = MaxwellState(ctx, mesh.num_elements, mesh.num_ghost)

    def fluid_ic(x: Real, y: Real) -> Array[Real, EULER_NC]:
        return [
            Real(1.0) + DELTA * sin(TWO_PI * x),
            Real(0.0), Real(0.0), Real(0.0),
            Real(1.0e-4) / (GAMMA - Real(1.0)),
        ]
    def maxwell_ic(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = (DELTA / TWO_PI) * cos(TWO_PI * x)
        return v^
    mesh.fill(q_fluid, fluid_ic)
    mesh.fill(q_maxwell, maxwell_ic)
    var mass_0 = mesh.integrate(q_fluid, IDX_RHO)

    var fluid_spatial = DGEulerSolver[2, P](Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    ))
    var maxwell_spatial = DGMaxwellSolver[2, P](
        Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    )
    var stepper = SSPRK3_TwoFluid[2, P](ctx, mesh.num_elements, mesh.num_ghost)

    def compute_joint_rhs(
        mut qf_in: FluidState, mut qm_in: MaxwellState,
        mut qf_out: FluidState, mut qm_out: MaxwellState,
    ) raises {imm}:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source(ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source(ctx, qf_in, qm_out, ALPHA, EPS0)

    var dt: Real = 5.0e-3
    var steps = Int(TWO_PI / dt)
    for _ in range(steps):
        stepper.step(ctx, q_fluid, q_maxwell, dt, compute_joint_rhs)

    def exact_rho(x: Real, y: Real) {imm} -> Real:
        return Real(1.0) + DELTA * cos(Real(steps) * dt) * sin(TWO_PI * x)
    var err = mesh.l2_error(q_fluid, IDX_RHO, exact_rho)
    var mass_1 = mesh.integrate(q_fluid, IDX_RHO)
    var sum_f = mesh.checksum(q_fluid)
    var sum_m = mesh.checksum(q_maxwell)
    if is_root():
        print("ranks =", num_ranks(), " L2 error in rho =", err,
              " mass drift =", mass_1 - mass_0)
        print("CHECKSUM fluid", sum_f)
        print("CHECKSUM maxwell", sum_m)
    if not (err < Real(0.05) * DELTA):
        raise Error("density did not return after one plasma period")
    if not (fabs(mass_1 - mass_0) < Real(1.0e-12)):
        raise Error("mass not conserved")
    if is_root():
        print("PASSED")
