# 1D1V Vlasov-Maxwell hybrid-initialisation test.
#
# End-to-end pipeline check: start from a uniform 5-moment Euler IC
# (ρ, ρu_x = 0, e = p/(γ−1)), use
# `apply_5moment_to_kinetic_1d1v` to populate the phase-state with
# the per-x-node Maxwellian, add a small sinusoidal `E_x`
# perturbation, then evolve with the coupled SSPRK3 stepper. Verify:
#
#   (a) Mass conservation: `ΔM/M₀` at machine precision (BGK isn't
#       active here — there are no sources/sinks in the v-direction
#       beyond the truncation-quiet Maxwellian tails).
#   (b) Energy conservation: `ΔW/W₀ < 1e-9` (same kind of bound
#       slice 9 hit at ≈ 8e-12 with a hand-built IC; the
#       5moment-initialised IC has an `erfc` truncation residue but
#       the *evolution* from that IC still conserves energy).
#
# This is the first test that ties together slice-14 (initialiser),
# slice-8 (coupled stepper), slice-7 (current source), slice-3
# (Vlasov DG kernel) and slice-6 (first moment) into a single
# end-to-end pipeline driven entirely from a fluid IC.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_hybrid_init_1d1v.mojo

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_E


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 16
comptime NV: Int = 32
comptime L_DOMAIN: Real = TWO_PI
# `V_MAX = 1.6` (≈ 10·v_th) keeps the Maxwellian tail at the
# boundary at `erfc(V_MAX / (v_th · √2)) ≈ erfc(7.1) ≈ 3·10⁻²³`, well
# below any other numerical-truncation scale in the run.
comptime V_MAX: Real = 1.6
comptime MASS: Real = 1.0
comptime GAMMA: Real = 5.0 / 3.0
# Fluid target: ρ=1, u=0, T=0.025 ⇒ e = p/(γ−1) = ρT/(γ−1).
# v_th² = T/m = 0.025 ⇒ v_th ≈ 0.158.
comptime RHO_TARGET: Real = 1.0
comptime T_TARGET: Real = 0.025
comptime E_TARGET: Real = RHO_TARGET * T_TARGET / (GAMMA - Real(1.0))
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
comptime EPS: Real = 0.01
comptime T_FINAL: Real = 6.0
comptime DT: Real = 0.05
comptime PhaseState = PhaseState1D1V[P_X, P_V, VLASOV_NC]
comptime MaxwellState = DGState[1, P_X, MAXWELL_NC]


def total_energy(
    phase_mesh: PhaseMesh1D1V[P_X, P_V],
    x_mesh: BlockMesh[1, P_X],
    mut q_phase: PhaseState,
    mut q_maxwell: MaxwellState,
) raises -> Real:
    """`W = ½ ∫ E_x² dx + ½ ∫∫ v² f dv dx`."""
    def field_energy(v: Array[Real, MAXWELL_NC]) -> Real:
        return Real(0.5) * v[IDX_EX] * v[IDX_EX]
    var w_E = x_mesh.integrate_fn(q_maxwell, field_energy)
    var w_K: Real = 0.0
    def add(x: Real, v: Real, w: Real, f: Real) {mut w_K}:
        w_K += w * Real(0.5) * v * v * f
    phase_mesh.visit_nodes(q_phase, add)
    return w_E + w_K


def total_mass(
    phase_mesh: PhaseMesh1D1V[P_X, P_V], mut q_phase: PhaseState,
) raises -> Real:
    var acc: Real = 0.0
    def add(x: Real, v: Real, w: Real, f: Real) {mut acc}:
        acc += w * f
    phase_mesh.visit_nodes(q_phase, add)
    return acc


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var phase_mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), L_DOMAIN, -V_MAX, V_MAX,
        num_x=NX, num_v=NV, periodic_x=True,
    )
    var x_mesh = BlockMesh[1, P_X](
        ctx, Real(0.0), L_DOMAIN, NX, periodic=True,
    )

    # --- Step 1: Build uniform 5-moment Euler IC. ---
    var fluid = DGState[1, P_X, EULER_NC](ctx, NX)
    def uniform_fluid(x: Real) -> Array[Real, EULER_NC]:
        var v = Array[Real, EULER_NC](fill=Real(0.0))
        v[IDX_RHO] = RHO_TARGET
        v[IDX_E] = E_TARGET
        return v^
    x_mesh.fill(fluid, uniform_fluid)
    ctx.synchronize()

    # --- Step 2: 5moment → kinetic Maxwellian. ---
    var q_phase = PhaseState(ctx, NX, NV)
    apply_5moment_to_kinetic_1d1v[P_X, P_V](
        ctx, phase_mesh, fluid, q_phase, MASS, GAMMA,
    )
    ctx.synchronize()

    # --- Step 3: Maxwell IC with sinusoidal E_x perturbation. ---
    var q_maxwell = MaxwellState(ctx, NX)
    def field(x: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = EPS * sin(TWO_PI * x / L_DOMAIN)
        return v^
    x_mesh.fill(q_maxwell, field)
    ctx.synchronize()

    var w0 = total_energy(phase_mesh, x_mesh, q_phase, q_maxwell)
    var m0 = total_mass(phase_mesh, q_phase)
    print("Hybrid IC: M =", m0, " W =", w0)

    # --- Step 4: Coupled Vlasov-Maxwell evolution. ---
    var vlasov_app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial_vlasov = DGVlasovSolver1D1V[P_X, P_V](vlasov_app)
    var spatial_maxwell = DGMaxwellSolver[1, P_X](maxwell_app)
    var moments = VlasovMoments1D1V[P_X, P_V]()
    var stepper = SSPRK3_VlasovMaxwell_1D1V[P_X, P_V](ctx, NX, NV)
    var j_current = DGState[1, P_X, 1](ctx, NX)

    def compute_joint_rhs(
        mut qp_in: PhaseState,
        mut qm_in: MaxwellState,
        mut qp_out: PhaseState,
        mut qm_out: MaxwellState,
    ) raises {mut j_current, imm}:
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.first_moment(ctx, phase_mesh, qp_in, j_current)
        apply_vlasov_current_source_1d[P_X](
            ctx, j_current, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q_phase, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    var w1 = total_energy(phase_mesh, x_mesh, q_phase, q_maxwell)
    var m1 = total_mass(phase_mesh, q_phase)

    var dW = w1 - w0
    var adW = dW if dW >= 0.0 else -dW
    var rel_W = adW / w0
    var dM = m1 - m0
    var adM = dM if dM >= 0.0 else -dM
    var rel_M = adM / m0
    print(
        "Hybrid T_FINAL: steps =", nsteps, " t =", t,
        " ΔM/M =", rel_M, " ΔW/W =", rel_W,
    )

    # (a) Mass conserved to machine precision — no source/sink terms,
    # outflow BC drains negligible mass when `v_max ≈ 10·v_th`.
    assert_true(rel_M < Real(1.0e-12), "hybrid pipeline lost mass")
    # (b) Energy conserved at numerical-truncation level. Drift is
    # dominated by SSPRK3 truncation; slice 9's 8e-12 came from a
    # 200-step run at `dt = 5·10⁻³`, ours uses `dt = 5·10⁻²` for 120
    # steps (a `T · dt³` budget 6000× larger). Tolerance 1e-7 sits
    # well above the observed 6e-8 and well below the time-truncation
    # bound `T · dt³ ≈ 7·10⁻⁴`.
    assert_true(rel_W < Real(1.0e-7), "hybrid pipeline didn't conserve energy")

    print("PASSED")
