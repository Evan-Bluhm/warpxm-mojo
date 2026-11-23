# Coupled Vlasov-Maxwell 1D1V — uniform plasma oscillator smoke test.
#
# A uniform-in-x Maxwellian distribution and a uniform-in-x initial
# `E_x` perturbation cannot develop any x-dependence — there are no
# spatial gradients to drive the spatial DG terms. The dynamics
# reduces to a closed ODE in the bulk fluid moments:
#
#     du/dt = α · E
#     dE/dt = -α · n · u / ε₀
#
# With `α = q/m = -1`, `n = ∫ f dv = 1` (we use a *normalised*
# Maxwellian so n=1), and `ε₀ = 1`, this is `u'' = -u`. Period `2π`.
#
# Exact solution for `E(0) = E0`, `u(0) = 0`:
#
#     E(t) = E0 · cos t,        J(t) = n·u(t) = -E0 · sin t.
#
# We integrate one full period with the coupled SSPRK3 stepper and
# compare `E_x` and the species current `J = ∫ v f dv` with that
# solution at every quarter period (t = π/2, π, 3π/2, 2π). Checking
# only the end of the period is not enough: with either half of the
# coupling removed `E_x` simply stays at `E0` and "returns" trivially.
# The quarter-period samples pin the frequency (E must vanish at π/2
# and 3π/2), the amplitude and sign of the particle response (J = ∓E0
# there), and the sign of the field response (E = -E0 at π).
#
# Also checked: `E_x` stays uniform in x to machine precision — no
# spurious spatial structure leaks in from the kernel.
#
# This is the smallest non-trivial test of the coupled stepper. A
# more ambitious Langmuir-oscillation test with a spatially varying
# perturbation (and Landau damping) is reserved for a later slice.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_uniform_oscillator_1d1v.mojo

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 16
comptime SIGMA_SQ: Real = 0.5
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
# n = ∫ exp(-v²/σ²)/(σ√π) dv ≈ 1 (with v_max=3, σ²=0.5 the truncation
# residue is ~ 1e-9). Plasma frequency in these normalised units is
# ω_p² = n · α² / ε₀ = 1, period = 2π.
comptime T_FINAL: Real = 2.0 * PI
# 80 steps per quarter period, so the samples land exactly on kπ/2.
comptime STEPS_PER_QUARTER: Int = 80
comptime DT: Real = T_FINAL / Real(4 * STEPS_PER_QUARTER)
comptime E0: Real = 0.01  # initial uniform field perturbation
comptime Phase = PhaseState[1, 1, P_X, P_V, VLASOV_NC]
comptime MaxwellState = DGState[1, P_X, MAXWELL_NC]
comptime MomentState = DGState[1, P_X, VLASOV_MOMENT_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # --- meshes ---
    # The phase mesh's x-axis and the Maxwell mesh must align (same NX,
    # same domain, same periodicity) so the aux buffer indexes match.
    var phase_mesh = PhaseMesh[1, 1, P_X, P_V](
        ctx,
        lo=[Real(0.0), Real(-3.0)],
        hi=[Real(1.0), Real(3.0)],
        n=[NX, NV],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P_X](
        ctx, Real(0.0), Real(1.0), NX, periodic=True,
    )

    # --- state ---
    var q_phase = Phase(ctx, phase_mesh.num_phase_elements)
    var q_maxwell = MaxwellState(ctx, NX)

    # IC: f₀(v) = exp(-v²/σ²) / (σ√π) — normalised so ∫ f dv ≈ 1.
    var sigma = sqrt(SIGMA_SQ)
    var norm = Real(1.0) / (sigma * sqrt(PI))
    def maxwellian(x: Real, v: Real) {imm} -> Real:
        return norm * exp(-v * v / SIGMA_SQ)
    phase_mesh.fill(q_phase, maxwellian)

    # IC: uniform E_x = E0, all other Maxwell components zero.
    def uniform_field(x: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = E0
        return v^
    x_mesh.fill(q_maxwell, uniform_field)
    ctx.synchronize()

    # --- solvers + stepper ---
    var vlasov_app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial_vlasov = DGVlasovSolver[1, 1, P_X, P_V](vlasov_app)
    var spatial_maxwell = DGMaxwellSolver[1, P_X](maxwell_app)
    var moments = VlasovMoments[1, 1, P_X, P_V](ctx, phase_mesh)
    var stepper = SSPRK3VlasovMaxwell[1, 1, P_X, P_V](
        ctx, phase_mesh.num_phase_elements, phase_mesh.num_x_elements,
    )

    # Scratch state for the velocity moments at each RHS evaluation; the
    # species current is its `MOM_FLUX_X` component.
    var mom = MomentState(ctx, NX)

    def compute_joint_rhs(
        mut qp_in: Phase,
        mut qm_in: MaxwellState,
        mut qp_out: Phase,
        mut qm_out: MaxwellState,
    ) raises {mut mom, imm}:
        # Vlasov reads E_x from qm_in (the Maxwell aux at this stage).
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        # Maxwell's spatial RHS (purely from its own state).
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        # Species current density `J = ∫ v f dv`.
        moments.compute(ctx, phase_mesh, qp_in, mom)
        # Add `-α/ε₀ · J` to `qm_out.E_x`. Closes the two-way coupling.
        apply_vlasov_current_source[1, 1, P_X](
            ctx, mom, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    # --- time loop: one full plasma period, sampled every quarter ---
    # Exact (E/E0, J/E0) at t = π/2, π, 3π/2, 2π.
    var e_exact = [Real(0.0), Real(-1.0), Real(0.0), Real(1.0)]
    var j_exact = [Real(-1.0), Real(0.0), Real(1.0), Real(0.0)]
    var max_e_err: Real = 0.0
    var max_j_err: Real = 0.0
    var max_x_spread: Real = 0.0
    var t: Real = 0.0
    for quarter in range(4):
        for _ in range(STEPS_PER_QUARTER):
            stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
            t += DT
        moments.compute(ctx, phase_mesh, q_phase, mom)
        ctx.synchronize()
        var e_node0: Real
        with q_maxwell.buf.map_to_host() as h:
            e_node0 = h[MaxwellState.index(0, 0, IDX_EX)]
        def check_e(
            x: Real, w: Real, v: Array[Real, MAXWELL_NC],
        ) {mut max_e_err, mut max_x_spread, imm}:
            max_e_err = fmax(max_e_err, abs(v[IDX_EX] - E0 * e_exact[quarter]))
            max_x_spread = fmax(max_x_spread, abs(v[IDX_EX] - e_node0))
        x_mesh.visit_nodes(q_maxwell, check_e)
        var j_node0: Real
        with mom.buf.map_to_host() as hj:
            j_node0 = hj[MomentState.index(0, 0, MOM_FLUX_X)]
        def check_j(
            x: Real, w: Real, v: Array[Real, VLASOV_MOMENT_NC],
        ) {mut max_j_err, imm}:
            max_j_err = fmax(
                max_j_err, abs(v[MOM_FLUX_X] - E0 * j_exact[quarter]),
            )
        x_mesh.visit_nodes(mom, check_j)
        print(
            "  t =", t, " E_x/E0 =", e_node0 / E0, " J/E0 =", j_node0 / E0,
        )

    print(
        "Vlasov-Maxwell oscillator 1D1V: max |E - E_exact| / E0 =",
        max_e_err / E0,
        " max |J - J_exact| / E0 =", max_j_err / E0,
        " x-spread =", max_x_spread,
    )

    # Measured: 2.0e-6 (E) and 1.5e-6 (J) relative to E0 — the SSPRK3
    # amplitude / phase error accumulated over one period.
    assert_true(
        max_e_err < Real(1.0e-5) * E0,
        "E_x does not follow E0·cos(ω_p t)",
    )
    assert_true(
        max_j_err < Real(1.0e-5) * E0,
        "species current does not follow -E0·sin(ω_p t)",
    )
    # E_x must stay uniform in x — the kernel correctly handles the
    # no-x-physics case. Machine precision.
    assert_true(
        max_x_spread < Real(1.0e-12),
        "E_x developed spurious x-structure",
    )

    print("PASSED")
