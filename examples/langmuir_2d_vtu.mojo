# 2D Langmuir oscillation — single-fluid plasma with VTU output.
#
# A mobile electron fluid against a static positive-ion background.
# A small density perturbation drives an E_x field that oscillates the
# electrons at the plasma frequency ω_pe = √(α²·n_0/ε_0) = 1.
#
# Setup (same as tests/test_langmuir_2d.mojo but extruded to a thin
# 2D slab so the field can be visualised in ParaView):
#   * 64×8 periodic mesh on [0,1] × [0, 1/8]
#   * α = -1, n_0 = 1, ε_0 = 1
#   * ρ_e(x, 0) = 1 + δ·sin(2π x), δ = 0.01
#   * E_x(x, 0) = (δ/(k·ε_0))·cos(2π x)
#
# Output: two parallel PVD collections per frame — one for the fluid
# state (`langmuir_fluid_NNNN.vtu`) and one for the Maxwell field
# (`langmuir_maxwell_NNNN.vtu`). Both share the same time stamps so
# ParaView can play them in sync.
#
# Time step: fixed for the run. The coupled (fluid, Maxwell) stepper has no
# CFL-driven variant, and here nothing changes the limit anyway: light
# (c = 1) is far faster than the cold electron fluid, so the Maxwell CFL
# step bounds the hyperbolic part, and it also resolves the plasma
# frequency that the source terms introduce (ω_pe · dt ≈ 1e-3).
#
#     pixi run example langmuir_2d_vtu
#     # then open output/langmuir/langmuir_*.pvd in ParaView

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.plasma import *


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 8
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = -1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime NUM_PERIODS: Real = 2.0           # number of Langmuir periods to integrate
comptime NUM_FRAMES: Int = 13              # frames 0 .. 12 — covers two periods
comptime OUTPUT_DIR: String = "output/langmuir"
comptime Fluid = DGState[2, P, EULER_NC]
comptime Field = DGState[2, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_fluid = Fluid(ctx, mesh.num_elements, mesh.num_ghost)
    var q_maxwell = Field(ctx, mesh.num_elements, mesh.num_ghost)

    def fluid_initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        return [
            Real(1.0) + DELTA * sin(TWO_PI * x), Real(0.0), Real(0.0), Real(0.0),
            P0 / (GAMMA - Real(1.0)),
        ]

    def field_initial(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = (DELTA / (TWO_PI * EPS0)) * cos(TWO_PI * x)
        return v^
    mesh.fill(q_fluid, fluid_initial)
    mesh.fill(q_maxwell, field_initial)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver[2, P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver[2, P](maxwell_app)
    var stepper = SSPRK3_TwoFluid[2, P](ctx, mesh.num_elements, mesh.num_ghost)

    var fw_fluid = FrameWriter("langmuir_fluid", OUTPUT_DIR)
    var fw_maxwell = FrameWriter("langmuir_maxwell", OUTPUT_DIR)

    var omega_p: Real = 1.0       # √(α²·n_0/ε_0) = 1
    var T = TWO_PI / omega_p      # Langmuir period ≈ 6.283
    var t_final = NUM_PERIODS * T
    var dt = fmin(
        maxwell_spatial.suggest_dt(mesh),
        fluid_spatial.suggest_dt(ctx, mesh, q_fluid),
    )

    def compute_joint_rhs(
        mut qf_in: Fluid, mut qm_in: Field, mut qf_out: Fluid, mut qm_out: Field,
    ) raises {imm}:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source(ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source(ctx, qf_in, qm_out, ALPHA, EPS0)

    def ex_squared(v: Array[Real, MAXWELL_NC]) -> Real:
        return v[IDX_EX] * v[IDX_EX]

    var mass_0 = mesh.integrate(q_fluid, IDX_RHO)
    root_print("2D Langmuir oscillation, ω_pe =", omega_p, ", T =", T, ", dt =", dt)
    root_print("time   ∫ρ_e dV   ∫E_x² dV   rel Δmass")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_euler_frame(fw_fluid.next_path(t), mesh, q_fluid, GAMMA)
        dump_maxwell_frame(fw_maxwell.next_path(t), mesh, q_maxwell)
        var mass = mesh.integrate(q_fluid, IDX_RHO)
        root_print(
            t, mass, mesh.integrate_fn(q_maxwell, ex_squared),
            (mass - mass_0) / mass_0,
        )

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until(
                ctx, q_fluid, q_maxwell,
                t, t_final * Real(frame + 1) / Real(NUM_FRAMES - 1), dt,
                compute_joint_rhs,
            )

    fw_fluid.write_pvd()
    fw_maxwell.write_pvd()
    root_print("Wrote", NUM_FRAMES, "fluid +", NUM_FRAMES, "Maxwell frames at", OUTPUT_DIR)
    root_print("Open langmuir_fluid.pvd or langmuir_maxwell.pvd in ParaView.")
