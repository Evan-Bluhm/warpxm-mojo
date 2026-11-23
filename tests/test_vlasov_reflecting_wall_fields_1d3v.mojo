# 1D3V specular walls (`BC_VLASOV_REFLECT`) with electric and magnetic
# fields. `test_vlasov_reflecting_wall_1d3v` covers the walls with free
# streaming only; here the particles are pushed into them.
#
# (1) Confined equilibrium. For a static `E = (E_x(x), 0, 0)` with
#     potential `Φ' = E_x` and ANY magnetic field `B(x)`, the energy
#     `H = |v|²/2 − κ Φ(x)` (`κ = (q/m)/λ`) is a constant of the motion,
#     so
#
#         f = exp(−H / T)
#
#     is a steady state of the Vlasov equation, and being even in `v_1`
#     it satisfies the specular condition at both walls. `E_x` here
#     pushes the particles into both walls and `B` has all three
#     components, the tangential ones turning `v_1` into `v_2`, `v_3`
#     right at the wall. The solution must stay put: the L² error at
#     `T_EQ` converges and no mass leaves through the walls. A wall that
#     lets particles leave drains the box instead.
#
# (2) Bouncing and gyrating. With `B = (B_x, 0, 0)` normal to the walls
#     and a tangential `E_y`, the force does not involve `v_1` or `x`, so
#     the motion along x is free streaming with reflection (the image
#     solution of the field-free test) while `(v_2, v_3)` gyrates about
#     the drift `v_3 = −E_y / B_x`:
#
#         f(x, v, t) = h(x − v_1 t, v_1) · g(R(−ωt) (v_2, v_3 + E_y/B_x)),
#
#     `ω = κ B_x`. `g` is an off-centre Gaussian, so a wall that mirrored
#     a tangential velocity as well as the normal one would be an O(1)
#     error.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_reflecting_wall_fields_1d3v.mojo

from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime P: Int = 2
comptime KAPPA: Real = 1.0          # (q/m)/λ
comptime State = PhaseState[1, 3, P, P, VLASOV_NC]
comptime Fields = DGState[1, P, MAXWELL_NC]

# --- (1) equilibrium ---
comptime EQ_V_MAX: Real = 6.0
comptime EQ_TEMP: Real = 1.0
comptime EQ_E_SLOPE: Real = 1.5     # E_x = EQ_E_SLOPE · (x − EQ_E_ZERO)
comptime EQ_E_ZERO: Real = 0.3
comptime T_EQ: Real = 0.3

# --- (2) bounce + gyration ---
comptime GY_V_MAX: Real = 5.0
comptime GY_NV1: Int = 8
comptime GY_BX: Real = 2.0
comptime GY_EY: Real = 0.6
comptime GY_EPS: Real = 0.3
comptime GY_DRIFT: Real = 0.8
comptime GY_INV_2SIGMA_SQ: Real = 1.0 / (2.0 * 0.7 * 0.7)
comptime GY_C2: Real = 0.5
comptime GY_C3: Real = -0.4
comptime T_GY: Real = 0.5


def eq_fields(x: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EX] = EQ_E_SLOPE * (x - EQ_E_ZERO)
    v[IDX_BX] = 0.4
    v[IDX_BY] = 0.5 * cos(PI * x)
    v[IDX_BZ] = -0.3 + 0.6 * x
    return v^


def eq_exact(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    var potential = EQ_E_SLOPE * (0.5 * x * x - EQ_E_ZERO * x)
    var energy = 0.5 * (v1 * v1 + v2 * v2 + v3 * v3) - KAPPA * potential
    return exp(-energy / EQ_TEMP)


def gy_fields(x: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EY] = GY_EY
    v[IDX_BX] = GY_BX
    return v^


def gy_exact_at(x: Real, v1: Real, v2: Real, v3: Real, t: Real) -> Real:
    # Along x: the image solution, evaluated at the foot of the
    # free-streaming characteristic.
    var y = x - v1 * t
    var w1 = v1 - GY_DRIFT * sin(PI * y)
    var along = (1.0 + GY_EPS * cos(PI * y)) * exp(-GY_INV_2SIGMA_SQ * w1 * w1)
    # In (v2, v3): rotate back about the E × B drift.
    var omega = KAPPA * GY_BX
    var w3 = v3 + GY_EY / GY_BX
    var c = cos(omega * t)
    var s = sin(omega * t)
    var v2_0 = v2 * c - w3 * s
    var v3_0 = v2 * s + w3 * c - GY_EY / GY_BX
    var d2 = v2_0 - GY_C2
    var d3 = v3_0 - GY_C3
    return along * exp(-GY_INV_2SIGMA_SQ * (d2 * d2 + d3 * d3))


def gy_initial(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    return gy_exact_at(x, v1, v2, v3, Real(0.0))


def gy_final(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    return gy_exact_at(x, v1, v2, v3, T_GY)


def wall_run[
    FieldsFn: def(Real) -> Array[Real, MAXWELL_NC],
    Initial: def(Real, Real, Real, Real) -> Real,
    Final: def(Real, Real, Real, Real) -> Real,
](
    var ctx: DeviceContext,
    label: String,
    nx: Int, nv1: Int, nv_perp: Int, v_max: Real, t_final: Real,
    bc_kind: Int, mass_tol: Real,
    fields_fn: FieldsFn, initial: Initial, final: Final,
) raises -> Real:
    """Run to `t_final` between two walls of kind `bc_kind` in static
    fields; assert the relative mass change is below `mass_tol` and
    return the relative L² error."""
    var mesh = PhaseMesh[1, 3, P, P](
        ctx,
        lo=[0.0, -v_max, -v_max, -v_max], hi=[1.0, v_max, v_max, v_max],
        n=[nx, nv1, nv_perp, nv_perp],
        periodic=[False], bc=[bc_kind, bc_kind],
    )
    var x_mesh = BlockMesh[1, P](
        ctx, 0.0, 1.0, nx, periodic=False,
        bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
    )
    mesh.check_companion(x_mesh)
    var em = Fields(ctx, nx)
    x_mesh.fill(em, fields_fn)
    var q = State(ctx, mesh.num_phase_elements)
    mesh.fill(q, initial)
    var mass0 = mesh.integrate(q)

    var spatial = DGVlasovSolver[1, 3, P, P](
        VlasovMaxwell(
            charge_mass_ratio=KAPPA, skin_depth_norm=Real(1.0),
            cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3Phase[1, 3, P, P, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, em, q_out)
    def no_adjust(mut q_: State) raises:
        pass
    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, em)
    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), t_final, compute_rhs, no_adjust, suggest_dt,
        dt_interval=1000000,   # the fields are static
    )
    ctx.synchronize()

    var dm = (mesh.integrate(q) - mass0) / mass0
    var err_sq: Real = 0.0
    var norm_sq: Real = 0.0
    def add(
        x: Real, v1: Real, v2: Real, v3: Real, w: Real, f: Real,
    ) {mut err_sq, mut norm_sq, imm}:
        var exact = final(x, v1, v2, v3)
        err_sq += w * (f - exact) * (f - exact)
        norm_sq += w * exact * exact
    mesh.visit_nodes(q, add)
    var err = sqrt(err_sq / norm_sq)
    print(
        " ", label, " n =", nx, nv1, nv_perp,
        " relative mass change =", dm, " L2 error =", err,
    )
    assert_true(abs(dm) < mass_tol, label + ": mass not conserved")
    return err


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # (1) Measured: error 1.38e-2 → 3.12e-3 (rate 2.15), mass change
    # 1.4e-9 and 1.7e-9. The mass change is the Maxwellian tail leaving
    # through the velocity boundary at |v| = 6, not the walls: the same
    # problem on a periodic x axis with no walls gives 1.4e-9 and 2.2e-9,
    # and a rate of 2.41 (a unit-width Maxwellian on elements of width 1
    # and 0.5 is short of the asymptotic range, hence the wide slack).
    # The mass is the sharp check: copy-out walls lose 5e-3 of it and
    # leave an error of 9.6e-3 on the fine mesh.
    var eq_lo = wall_run(
        ctx, "equilibrium", 8, 12, 12, EQ_V_MAX, T_EQ,
        BC_VLASOV_REFLECT, Real(1.0e-7), eq_fields, eq_exact, eq_exact,
    )
    var eq_hi = wall_run(
        ctx, "equilibrium", 16, 24, 24, EQ_V_MAX, T_EQ,
        BC_VLASOV_REFLECT, Real(1.0e-7), eq_fields, eq_exact, eq_exact,
    )
    _ = check_convergence(
        "1D3V wall equilibrium P = 2 (8 × 12³ → 16 × 24³)",
        eq_lo, eq_hi, Real(3.0), Real(0.02), Real(0.0047),
        rate_slack=Real(1.0),
    )

    # (2) Measured: error 4.81e-2 → 6.99e-3 (rate 2.78), mass change
    # 1.8e-7 and 1.1e-9 (velocity boundary again; the gyration carries
    # the off-centre Gaussian towards it). With copy-out walls the error
    # is 0.44 at both resolutions.
    var gy_lo = wall_run(
        ctx, "bounce + gyration", 16, GY_NV1, 12, GY_V_MAX, T_GY,
        BC_VLASOV_REFLECT, Real(1.0e-6), gy_fields, gy_initial, gy_final,
    )
    var gy_hi = wall_run(
        ctx, "bounce + gyration", 32, GY_NV1, 24, GY_V_MAX, T_GY,
        BC_VLASOV_REFLECT, Real(1.0e-6), gy_fields, gy_initial, gy_final,
    )
    _ = check_convergence(
        "1D3V wall bounce + gyration P = 2 (16 × 12² → 32 × 24²)",
        gy_lo, gy_hi, Real(3.0), Real(0.072), Real(0.0105),
        rate_slack=Real(0.35),
    )
    print("PASSED")
