# Steady-state stepping driver for nsys / ncu profiling of the 1D2V
# Vlasov-Maxwell hot path (kinetic RHS, velocity moments, current source,
# Maxwell RHS, SSPRK3 combines). Same setup as
# `tests/test_vlasov_maxwell_weibel_growth_1d2v.mojo` on a mesh large
# enough to fill the GPU: 64 x 32 x 32 phase-space elements at P = 2
# (1.77M nodes). Prints steps/s over the measured window.
#
#     pixi run mojo run -I src profiling/prof_vlasov_1d2v.mojo

from std.collections import Array
from std.math import exp, sin
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.maxwell import *
from warpxm.physics.vlasov import *


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 64
comptime NV1: Int = 32
comptime NV2: Int = 32
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = TWO_PI
comptime V1_MAX: Real = 1.5     # 5 σ_par
comptime V2_MAX: Real = 4.0     # 4 σ_perp
comptime SIGMA_PAR_SQ: Real = 0.09
comptime SIGMA_PERP_SQ: Real = 1.0
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
# Growing root of the kinetic dispersion relation (see header).
comptime GAMMA_W: Real = 0.364360
comptime N_WARMUP: Int = 5
comptime N_STEPS: Int = 30
comptime DT: Real = 5.0e-3
comptime DELTA: Real = 1.0e-6       # B_z seed amplitude
comptime SAMPLE_EVERY: Int = 4      # sample stride for the B_z amplitude
comptime FIT_T_MIN: Real = 10.0     # linear-fit window
comptime FIT_T_MAX: Real = 20.0
comptime Phase = PhaseState[1, 2, P_X, P_V, VLASOV_NC]
comptime MaxwellState = DGState[1, P_X, MAXWELL_NC]


def compute_total_mass(
    phase_mesh: PhaseMesh[1, 2, P_X, P_V], mut q_phase: Phase,
) raises -> Real:
    var acc: Real = 0.0
    def add(x: Real, v1: Real, v2: Real, w: Real, f: Real) {mut acc}:
        acc += w * f
    phase_mesh.visit_nodes(q_phase, add)
    return acc


def bz_sin_amplitude(
    x_mesh: BlockMesh[1, P_X], mut q_maxwell: MaxwellState,
) raises -> Real:
    """Amplitude of the sin(kx) Fourier mode of B_z: (2/L) ∫ B_z sin(kx) dx."""
    var k_wave = TWO_PI / (X_MAX - X_MIN)
    var acc: Real = 0.0
    def add(x: Real, w: Real, v: Array[Real, MAXWELL_NC]) {mut acc, imm}:
        acc += w * v[IDX_BZ] * sin(k_wave * x)
    x_mesh.visit_nodes(q_maxwell, add)
    return Real(2.0) * acc / (X_MAX - X_MIN)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var phase_mesh = PhaseMesh[1, 2, P_X, P_V](
        ctx,
        lo=[X_MIN, -V1_MAX, -V2_MAX],
        hi=[X_MAX, V1_MAX, V2_MAX],
        n=[NX, NV1, NV2],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P_X](
        ctx, X_MIN, X_MAX, NX, periodic=True,
    )

    var q_phase = Phase(ctx, phase_mesh.num_phase_elements)
    var q_maxwell = MaxwellState(ctx, NX)

    var k_wave = TWO_PI / (X_MAX - X_MIN)
    # IC: f = f₀ = norm · exp(-v1²/σ_par² - v2²/σ_perp²), n = 1 ⇒
    # norm = 1/(π σ_par σ_perp). No δf: the B_z seed alone projects onto
    # the growing mode.
    var sigma_par = sqrt(SIGMA_PAR_SQ)
    var sigma_perp = sqrt(SIGMA_PERP_SQ)
    var norm = Real(1.0) / (PI * sigma_par * sigma_perp)
    def f_initial(x: Real, v1: Real, v2: Real) {imm} -> Real:
        return norm * exp(
            -v1 * v1 / SIGMA_PAR_SQ
            - v2 * v2 / SIGMA_PERP_SQ
        )
    phase_mesh.fill(q_phase, f_initial)

    # IC: B_z(x, 0) = δ sin(kx), all other field components zero.
    def field_initial(x: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_BZ] = DELTA * sin(k_wave * x)
        return v^
    x_mesh.fill(q_maxwell, field_initial)
    ctx.synchronize()

    # Initial diagnostics.
    var m0 = compute_total_mass(phase_mesh, q_phase)
    var bz0 = bz_sin_amplitude(x_mesh, q_maxwell)
    print("Weibel IC: M =", m0, " B_z sin(kx) amplitude =", bz0)

    # --- solvers + stepper ---
    var vlasov_app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial_vlasov = DGVlasovSolver[1, 2, P_X, P_V](vlasov_app)
    var spatial_maxwell = DGMaxwellSolver[1, P_X](maxwell_app)
    var moments = VlasovMoments[1, 2, P_X, P_V](ctx, phase_mesh)
    var stepper = SSPRK3VlasovMaxwell[1, 2, P_X, P_V](
        ctx, phase_mesh.num_phase_elements, phase_mesh.num_x_elements,
    )
    var mom = DGState[1, P_X, VLASOV_MOMENT_NC](ctx, NX)

    def compute_joint_rhs(
        mut qp_in: Phase,
        mut qm_in: MaxwellState,
        mut qp_out: Phase,
        mut qm_out: MaxwellState,
    ) raises {mut mom, imm}:
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.compute(ctx, phase_mesh, qp_in, mom)
        apply_vlasov_current_source[1, 2, P_X](
            ctx, mom, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    for _ in range(N_WARMUP):
        stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
    ctx.synchronize()
    var elapsed_s = Real(perf_counter_ns() - t0) * Real(1.0e-9)
    var m1 = compute_total_mass(phase_mesh, q_phase)
    print(
        "BENCH config=vlasov_1d2v steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(Real(N_STEPS) / elapsed_s)
    )
    print("mass drift:", m1 - m0)
