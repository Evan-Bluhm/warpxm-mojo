# Time-dependent 2D3V and 3D3V runs: free streaming along every
# configuration axis, and a coupled Vlasov-Maxwell plasma oscillation.
# `test_vlasov_rhs_exact_3d3v` checks one evaluation of the right-hand
# side; these check that the steppers, moments and current coupling work
# in five and six dimensions.
#
# (1) Free streaming, 3D3V. With no fields `f(x, v, t) = f₀(x − v t, v)`.
#     `f₀` varies along an oblique wave vector with a different period on
#     each axis, so streaming along x, y and z each contribute. Velocity
#     nodes do not couple, so only the configuration mesh is refined and
#     the error converges at order `P + 1`.
#
# (2) Uniform plasma oscillation, 2D3V and 3D3V. A Maxwellian of unit
#     density drifting at `u₀` with `E = 0` oscillates rigidly: with
#     `q/m = −1`, `λ = ε₀ = 1`
#
#         du/dt = −E,   dE/dt = u    ⇒    u = u₀ cos t,   E = u₀ sin t.
#
#     `u₀` has three different components, so the three flux moments,
#     their coupling into Ampère's law and the three electric
#     accelerations are each checked with their own amplitude; a swapped
#     or missing component is an O(1) error. `B` must stay zero.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_oscillation_3d3v.mojo

from std.collections import Array
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime P: Int = 2
comptime V_MAX: Real = 4.0
comptime SIGMA_SQ: Real = 0.5

# --- (1) streaming ---
comptime ST_NV: Int = 3
comptime ST_EPS: Real = 0.4
comptime T_STREAM: Real = 0.15

# --- (2) oscillation ---
comptime OSC_NV: Int = 10
comptime U0_X: Real = 0.30
comptime U0_Y: Real = -0.20
comptime U0_Z: Real = 0.12
comptime T_OSC: Real = 2.0


def stream_exact(
    x: Real, y: Real, z: Real, v1: Real, v2: Real, v3: Real, t: Real,
) -> Real:
    var phase = TWO_PI * (
        (x - v1 * t) + 2.0 * (y - v2 * t) / 1.5 - (z - v3 * t) / 0.75
    )
    return (1.0 + ST_EPS * sin(phase)) * exp(
        -(v1 * v1 + v2 * v2 + v3 * v3) / SIGMA_SQ
    )


def stream_error(var ctx: DeviceContext, n: Int) raises -> Real:
    """Relative L² error of 3D3V free streaming at `T_STREAM` on an
    `n × n × n` configuration mesh."""
    comptime State = PhaseState[3, 3, P, P, VLASOV_NC]
    var mesh = PhaseMesh[3, 3, P, P](
        ctx,
        lo=[0.0, 0.0, 0.0, -V_MAX, -V_MAX, -V_MAX],
        hi=[1.0, 1.5, 0.75, V_MAX, V_MAX, V_MAX],
        n=[n, n, n, ST_NV, ST_NV, ST_NV],
        periodic=[True, True, True],
    )
    var em = DGState[3, P, MAXWELL_NC](ctx, mesh.num_x_elements)   # zero
    var q = State(ctx, mesh.num_phase_elements)
    def initial(
        x: Real, y: Real, z: Real, v1: Real, v2: Real, v3: Real,
    ) -> Real:
        return stream_exact(x, y, z, v1, v2, v3, Real(0.0))
    mesh.fill(q, initial)

    var spatial = DGVlasovSolver[3, 3, P, P](VlasovMaxwell(cfl_max=Real(0.5)))
    var stepper = SSPRK3Phase[3, 3, P, P, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, em, q_out)
    def no_adjust(mut q_: State) raises:
        pass
    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, em)
    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), T_STREAM, compute_rhs, no_adjust, suggest_dt,
        dt_interval=1000000,
    )

    var err_sq: Real = 0.0
    var norm_sq: Real = 0.0
    def add(
        x: Real, y: Real, z: Real, v1: Real, v2: Real, v3: Real,
        w: Real, f: Real,
    ) {mut err_sq, mut norm_sq}:
        var exact = stream_exact(x, y, z, v1, v2, v3, T_STREAM)
        err_sq += w * (f - exact) * (f - exact)
        norm_sq += w * exact * exact
    mesh.visit_nodes(q, add)
    return sqrt(err_sq / norm_sq)


def oscillation[XDIM: Int](var ctx: DeviceContext) raises -> Array[Real, 3]:
    """Run the drifting Maxwellian to `T_OSC` in `XDIM`D3V. Returns
    `[max error of E, max error of the flux, max |B|]`, the first two
    relative to `|u₀|`."""
    comptime Phase = PhaseState[XDIM, 3, P, P, VLASOV_NC]
    comptime Fields = DGState[XDIM, P, MAXWELL_NC]
    comptime ND = XDIM + 3
    var lo = Array[Real, ND](fill=-V_MAX)
    var hi = Array[Real, ND](fill=V_MAX)
    var n = Array[Int, ND](fill=OSC_NV)
    var x_counts = [2, 3, 2]
    var x_lengths = [1.0, 1.5, 0.75]
    for a in range(XDIM):
        lo[a] = 0.0
        hi[a] = x_lengths[a]
        n[a] = x_counts[a]
    var mesh = PhaseMesh[XDIM, 3, P, P](
        ctx, lo=lo, hi=hi, n=n, periodic=Array[Bool, XDIM](fill=True),
    )
    var block_hi = [Real(1.0), Real(1.0), Real(1.0)]
    var block_n = [1, 1, 1]
    for a in range(XDIM):
        block_hi[a] = x_lengths[a]
        block_n[a] = x_counts[a]
    var x_mesh = BlockMesh[XDIM, P](
        ctx,
        lo=[Real(0.0), Real(0.0), Real(0.0)], hi=block_hi, n=block_n,
        periodic=[True, True, True],
        bc=Array[Int, 6](fill=BC_COPYOUT),
    )
    mesh.check_companion(x_mesh)

    var q_phase = Phase(ctx, mesh.num_phase_elements)
    var q_maxwell = Fields(ctx, mesh.num_x_elements)   # E = B = 0
    var norm = Real(1.0) / (PI * SIGMA_SQ * sqrt(PI * SIGMA_SQ))
    def drifting(v1: Real, v2: Real, v3: Real) {imm} -> Real:
        var w1 = v1 - U0_X
        var w2 = v2 - U0_Y
        var w3 = v3 - U0_Z
        return norm * exp(-(w1 * w1 + w2 * w2 + w3 * w3) / SIGMA_SQ)
    comptime if XDIM == 2:
        def ic2(x: Real, y: Real, v1: Real, v2: Real, v3: Real) {imm} -> Real:
            return drifting(v1, v2, v3)
        mesh.fill(q_phase, ic2)
    else:
        def ic3(
            x: Real, y: Real, z: Real, v1: Real, v2: Real, v3: Real,
        ) {imm} -> Real:
            return drifting(v1, v2, v3)
        mesh.fill(q_phase, ic3)

    var spatial_vlasov = DGVlasovSolver[XDIM, 3, P, P](
        VlasovMaxwell(
            charge_mass_ratio=Real(-1.0), skin_depth_norm=Real(1.0),
            cfl_max=Real(0.5),
        )
    )
    var spatial_maxwell = DGMaxwellSolver[XDIM, P](
        Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    )
    var moments = VlasovMoments[XDIM, 3, P, P](ctx, mesh)
    var mom = DGState[XDIM, P, VLASOV_MOMENT_NC](ctx, mesh.num_x_elements)
    var stepper = SSPRK3VlasovMaxwell[XDIM, 3, P, P](
        ctx, mesh.num_phase_elements, mesh.num_x_elements,
    )

    def compute_joint_rhs(
        mut qp_in: Phase, mut qm_in: Fields,
        mut qp_out: Phase, mut qm_out: Fields,
    ) raises {mut mom, imm}:
        spatial_vlasov.compute_rhs(ctx, mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.compute(ctx, mesh, qp_in, mom)
        apply_vlasov_current_source[XDIM, 3, P](
            ctx, mom, qm_out, Real(-1.0), Real(1.0),
        )

    def suggest_dt(mut qp: Phase, mut qm: Fields) raises {imm} -> Real:
        return min(
            spatial_vlasov.suggest_dt(ctx, mesh, qm),
            spatial_maxwell.suggest_dt(x_mesh),
        )

    _ = stepper.run_until_cfl(
        ctx, q_phase, q_maxwell, Real(0.0), T_OSC,
        compute_joint_rhs, suggest_dt, dt_interval=20,
    )
    moments.compute(ctx, mesh, q_phase, mom)

    var u0 = [U0_X, U0_Y, U0_Z]
    var u0_norm = sqrt(U0_X * U0_X + U0_Y * U0_Y + U0_Z * U0_Z)
    var out = Array[Real, 3](fill=Real(0.0))
    with q_maxwell.buf.map_to_host() as h:
        for node in range(q_maxwell.total_nodes):
            for d in range(3):
                out[0] = max(
                    out[0],
                    abs(h[node * MAXWELL_NC + IDX_EX + d] - u0[d] * sin(T_OSC))
                    / u0_norm,
                )
                out[2] = max(out[2], abs(h[node * MAXWELL_NC + IDX_BX + d]))
    with mom.buf.map_to_host() as h:
        for node in range(mom.total_nodes):
            for d in range(3):
                out[1] = max(
                    out[1],
                    abs(
                        h[node * VLASOV_MOMENT_NC + MOM_FLUX_X + d]
                        - u0[d] * cos(T_OSC)
                    ) / u0_norm,
                )
    return out^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # Measured: 3.46e-3 → 4.78e-4, rate 2.85.
    var lo = stream_error(ctx, 4)
    var hi = stream_error(ctx, 8)
    _ = check_convergence(
        "3D3V free streaming P = 2 (4³ → 8³)",
        lo, hi, Real(3.0), Real(5.2e-3), Real(7.2e-4),
    )

    # Measured, the same in 2D3V and 3D3V to five digits: E within 1.4e-3
    # and the flux within 6.7e-4 of |u₀| (the velocity mesh, 10 elements
    # of width 0.8 per axis, resolving the shifted Maxwellian); |B| at
    # 1e-16. A component with the wrong sign or index is off by its
    # whole amplitude, at least 0.3 of |u₀|.
    var osc2 = oscillation[2](ctx)
    print(
        "2D3V oscillation: max |E − u₀ sin t|/|u₀| =", osc2[0],
        " max |Γ − u₀ cos t|/|u₀| =", osc2[1], " max |B| =", osc2[2],
    )
    var osc3 = oscillation[3](ctx)
    print(
        "3D3V oscillation: max |E − u₀ sin t|/|u₀| =", osc3[0],
        " max |Γ − u₀ cos t|/|u₀| =", osc3[1], " max |B| =", osc3[2],
    )
    assert_true(
        osc2[0] < Real(4.0e-3) and osc3[0] < Real(4.0e-3),
        "E departs from u₀ sin t",
    )
    assert_true(
        osc2[1] < Real(2.0e-3) and osc3[1] < Real(2.0e-3),
        "flux departs from u₀ cos t",
    )
    assert_true(
        osc2[2] < Real(1.0e-12) and osc3[2] < Real(1.0e-12),
        "a magnetic field appeared",
    )
    print("PASSED")
