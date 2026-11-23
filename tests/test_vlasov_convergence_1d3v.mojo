# 1D3V Vlasov convergence in time-dependent problems with exact solutions.
#
# (a) Free streaming. `E = B = 0`, so `∂f/∂t + v1 ∂f/∂x = 0` and
#     `f(x, v, T) = f₀(x − v1 T, v)`. The x mesh is refined at a fixed
#     velocity mesh; the L² error converges at order `P_x + 1`.
#
# (b) Gyration with E × B drift and parallel acceleration. `f` is uniform
#     in x and the fields are uniform, with `B` oblique to all three axes
#     and `E` neither parallel nor perpendicular to it. Each particle obeys
#     `dv/dt = α (E + v × B)`, whose solution is
#
#         v(t) = v_E + α E_∥ t b + R(−α|B|t) (v(0) − v_E),
#
#     `b = B/|B|`, `v_E = E × B / |B|²`, `E_∥ = E · b`, and `R(θ)` the
#     rotation about `b` by `θ`. So `f(v, T) = f₀(v(0))` with the map
#     inverted. Every one of the nine Lorentz terms is active and the
#     three velocity axes are mixed by the rotation; the velocity mesh is
#     refined and the L² error converges at order `P_v + 1`. The step
#     comes from `suggest_dt`.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_convergence_1d3v.mojo

from std.collections import Array
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0

# --- (a) free streaming ----------------------------------------------------
comptime STREAM_V_MAX: Real = 3.0
comptime STREAM_NV: Int = 4
comptime STREAM_SIGMA_SQ: Real = 0.5
comptime STREAM_T: Real = 0.1
comptime STREAM_DT: Real = 2.0e-4


def stream_initial(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    return sin(TWO_PI * x) * exp(-(v1 * v1 + v2 * v2 + v3 * v3) / STREAM_SIGMA_SQ)


def stream_error[P_X: Int](var ctx: DeviceContext, nx: Int) raises -> Real:
    comptime P_V = 2
    comptime State = PhaseState[1, 3, P_X, P_V, VLASOV_NC]
    var mesh = PhaseMesh[1, 3, P_X, P_V](
        ctx,
        lo=[X_MIN, -STREAM_V_MAX, -STREAM_V_MAX, -STREAM_V_MAX],
        hi=[X_MAX, STREAM_V_MAX, STREAM_V_MAX, STREAM_V_MAX],
        n=[nx, STREAM_NV, STREAM_NV, STREAM_NV],
        periodic=[True],
    )
    var q = State(ctx, mesh.num_phase_elements)
    mesh.fill(q, stream_initial)
    var spatial = DGVlasovSolver[1, 3, P_X, P_V](VlasovMaxwell())
    var stepper = SSPRK3Phase[1, 3, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    var em = DGState[1, P_X, MAXWELL_NC](ctx, nx)   # zero fields

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, em, q_out)
    def no_adjust(mut q_: State) raises:
        pass
    _ = stepper.run_until(
        ctx, q, Real(0.0), STREAM_T, STREAM_DT, compute_rhs, no_adjust,
    )
    ctx.synchronize()

    var acc: Real = 0.0
    def add(x: Real, v1: Real, v2: Real, v3: Real, w: Real, f: Real) {mut acc}:
        var period = X_MAX - X_MIN
        var x0 = x - v1 * STREAM_T
        while x0 < X_MIN:
            x0 += period
        while x0 >= X_MAX:
            x0 -= period
        var d = f - stream_initial(x0, v1, v2, v3)
        acc += w * d * d
    mesh.visit_nodes(q, add)
    return sqrt(acc)


# --- (b) gyration + E × B drift + parallel acceleration ---------------------
comptime GYRO_V_MAX: Real = 5.0
comptime GYRO_ALPHA: Real = 1.0          # an ion: q/m > 0
comptime GYRO_T: Real = 1.5              # |α B| T = 1.5 rad
comptime GYRO_INV_2SIGMA_SQ: Real = 1.0 / (2.0 * 0.6 * 0.6)
# B = (1, 2, 2)/3 (unit magnitude), E = (0.3, −0.2, 0.25).
comptime BX0: Real = 1.0 / 3.0
comptime BY0: Real = 2.0 / 3.0
comptime BZ0: Real = 2.0 / 3.0
comptime EX0: Real = 0.3
comptime EY0: Real = -0.2
comptime EZ0: Real = 0.25
comptime U0X: Real = 0.6
comptime U0Y: Real = -0.3
comptime U0Z: Real = 0.4


def gyro_initial(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    var w1 = v1 - U0X
    var w2 = v2 - U0Y
    var w3 = v3 - U0Z
    return exp(-GYRO_INV_2SIGMA_SQ * (w1 * w1 + w2 * w2 + w3 * w3))


def gyro_exact(x: Real, v1: Real, v2: Real, v3: Real, t: Real) -> Real:
    """`f₀` at the velocity the particle now at `v` started from."""
    # |B| = 1, so b = B.
    var e_par = EX0 * BX0 + EY0 * BY0 + EZ0 * BZ0
    # v_E = E × B / |B|²
    var dx = EY0 * BZ0 - EZ0 * BY0
    var dy = EZ0 * BX0 - EX0 * BZ0
    var dz = EX0 * BY0 - EY0 * BX0
    # w = v − v_E − α E_∥ t b
    var w1 = v1 - dx - GYRO_ALPHA * e_par * t * BX0
    var w2 = v2 - dy - GYRO_ALPHA * e_par * t * BY0
    var w3 = v3 - dz - GYRO_ALPHA * e_par * t * BZ0
    # Undo the rotation: rotate w about b by +α|B|t (Rodrigues).
    var theta = GYRO_ALPHA * t
    var c = cos(theta)
    var s = sin(theta)
    var b_dot_w = BX0 * w1 + BY0 * w2 + BZ0 * w3
    var cx = BY0 * w3 - BZ0 * w2
    var cy = BZ0 * w1 - BX0 * w3
    var cz = BX0 * w2 - BY0 * w1
    var r1 = w1 * c + cx * s + BX0 * b_dot_w * (1.0 - c)
    var r2 = w2 * c + cy * s + BY0 * b_dot_w * (1.0 - c)
    var r3 = w3 * c + cz * s + BZ0 * b_dot_w * (1.0 - c)
    return gyro_initial(x, r1 + dx, r2 + dy, r3 + dz)


def uniform_fields(x: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EX] = EX0
    v[IDX_EY] = EY0
    v[IDX_EZ] = EZ0
    v[IDX_BX] = BX0
    v[IDX_BY] = BY0
    v[IDX_BZ] = BZ0
    return v^


def gyro_error[P_V: Int](var ctx: DeviceContext, nv: Int) raises -> Real:
    comptime P_X = 1
    comptime NX = 2
    comptime State = PhaseState[1, 3, P_X, P_V, VLASOV_NC]
    var mesh = PhaseMesh[1, 3, P_X, P_V](
        ctx,
        lo=[X_MIN, -GYRO_V_MAX, -GYRO_V_MAX, -GYRO_V_MAX],
        hi=[X_MAX, GYRO_V_MAX, GYRO_V_MAX, GYRO_V_MAX],
        n=[NX, nv, nv, nv],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=True)
    var em = DGState[1, P_X, MAXWELL_NC](ctx, NX)
    x_mesh.fill(em, uniform_fields)

    var q = State(ctx, mesh.num_phase_elements)
    mesh.fill(q, gyro_initial)
    var mass0 = mesh.integrate(q)

    var spatial = DGVlasovSolver[1, 3, P_X, P_V](
        VlasovMaxwell(
            charge_mass_ratio=GYRO_ALPHA, skin_depth_norm=Real(1.0),
            cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3Phase[1, 3, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, em, q_out)
    def no_adjust(mut q_: State) raises:
        pass
    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, em)
    # The fields are static, so one suggestion serves the whole run.
    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), GYRO_T, compute_rhs, no_adjust, suggest_dt,
        dt_interval=1000000,
    )
    ctx.synchronize()

    var mass1 = mesh.integrate(q)
    print("  NV =", nv, " relative mass change =", (mass1 - mass0) / mass0)
    # Measured 5.2e-8 (NV = 16) and −2.5e-10 (NV = 32): what crosses the
    # velocity boundary, where f₀ is about 1e-9 of its peak.
    assert_true(
        abs(mass1 - mass0) < Real(1.0e-6) * mass0, "gyration lost mass",
    )

    var err_sq: Real = 0.0
    var norm_sq: Real = 0.0
    def add(
        x: Real, v1: Real, v2: Real, v3: Real, w: Real, f: Real,
    ) {mut err_sq, mut norm_sq}:
        var exact = gyro_exact(x, v1, v2, v3, GYRO_T)
        err_sq += w * (f - exact) * (f - exact)
        norm_sq += w * exact * exact
    mesh.visit_nodes(q, add)
    return sqrt(err_sq / norm_sq)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # Tolerances are about 1.5x to 3x the measured errors: streaming
    # 2.63e-4 → 3.32e-5 (rate 2.99), gyration 4.21e-2 → 5.97e-3 (rate 2.82;
    # the Gaussian is only a few elements wide on the coarse mesh).
    var s_lo = stream_error[2](ctx, 16)
    var s_hi = stream_error[2](ctx, 32)
    _ = check_convergence(
        "1D3V free streaming P_x = 2 (NX 16 → 32)",
        s_lo, s_hi, Real(3.0), Real(8.0e-4), Real(1.0e-4),
    )

    var g_lo = gyro_error[2](ctx, 16)
    var g_hi = gyro_error[2](ctx, 32)
    _ = check_convergence(
        "1D3V gyration P_v = 2 (NV 16 → 32)",
        g_lo, g_hi, Real(3.0), Real(0.065), Real(0.009),
        rate_slack=Real(0.35),
    )
    print("PASSED")
