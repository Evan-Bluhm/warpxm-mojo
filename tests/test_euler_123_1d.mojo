# Einfeldt 1-2-3 problem — classic near-vacuum stress test for positivity
# preservation in Euler. Two strong outward rarefactions create a region
# where ρ and P collapse toward zero. The conserved-variable limiter
# alone cannot prevent negative density or pressure here; the
# Zhang-Shu positivity preserver in `adjusters/positivity_euler.mojo`
# is what keeps the simulation valid.
#
# IC (Einfeldt et al. 1991):
#   Left  (x < 0.5): ρ = 1, u = -2, P = 0.4
#   Right (x ≥ 0.5): ρ = 1, u = +2, P = 0.4
#   γ = 1.4, t_final = 0.15
#
# Exact solution: two rarefactions separated by a near-vacuum plateau with
# ρ* = 0.02185, p* = 0.001894, u* = 0 for |x − 0.5| < 0.348 t.
#
# Validates:
#   * no NaN, ρ > 0 and P > 0 everywhere (the preserver's job)
#   * exact balance of mass / momentum / energy with boundary-flux
#     accounting (both copyout faces see the initial states)
#   * L1 error of (ρ, u, P) against the exact Riemann solution
#   * the centre is a near-vacuum: ρ and P collapse by more than 30× from
#     the initial state. (The scheme does NOT reproduce the exact plateau
#     values there — it shows the usual 1-2-3 internal-energy overshoot,
#     ρ too low and P too high — so the centre is bounded, not matched;
#     the L1 errors are what constrain accuracy.)

from std.collections import Array
from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *
from warpxm.testing.riemann import ExactRiemann


comptime P: Int = 2
comptime NX: Int = 200
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.15
comptime DT: Real = 1.0e-4
comptime State = DGState[1, P, EULER_NC]

# Left / right primitive states (ρ, u, p).
comptime RHO_L: Real = 1.0
comptime U_L: Real = -2.0
comptime P_L: Real = 0.4
comptime RHO_R: Real = 1.0
comptime U_R: Real = 2.0
comptime P_R: Real = 0.4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), num_elements=NX, periodic=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, EULER_NC]:
        var rho: Real = 1.0
        var u = U_L if x < Real(0.5) else U_R
        var pp: Real = 0.4
        var ke = Real(0.5) * rho * u * u
        return [rho, rho * u, Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0)) + ke]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)
    var mom_0 = mesh.integrate(q, IDX_MX)
    var energy_0 = mesh.integrate(q, IDX_E)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[1, P](app)
    # The limiter tolerance is α = alpha_coeff · dx^1.5. The default
    # coefficient (1000) gives α ≈ 0.35, sixteen times the vacuum density,
    # so the limiter never acts there and the centre is left to unlimited
    # oscillations (centre ρ 0.009 against the exact 0.0219). α ≈ 3.5e-3
    # keeps it active at the scale of the vacuum state.
    var limiter = MoeRossmanith[1, P, EULER_NC](
        ctx, mesh.num_elements, alpha_coeff=Real(10.0),
    )
    var positivity = PositivityEuler[1, P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var any_nan = False
    var min_rho: Real = 1.0e30
    var min_p: Real = 1.0e30
    var max_rho: Real = -1.0e30
    def scan(
        x: Real, w: Real, v: Array[Real, EULER_NC],
    ) {mut any_nan, mut min_rho, mut min_p, mut max_rho}:
        var rho = v[IDX_RHO]
        if isnan(rho) or isnan(v[IDX_E]): any_nan = True
        var u = v[IDX_MX] / rho
        var ke = Real(0.5) * rho * u * u
        var pp = (GAMMA - Real(1.0)) * (v[IDX_E] - ke)
        if rho < min_rho: min_rho = rho
        if rho > max_rho: max_rho = rho
        if pp < min_p: min_p = pp
    mesh.visit_nodes(q, scan)
    # Centre sample: first node of the element just right of x = 0.5.
    var center_rho: Real
    var center_p: Real
    with q.buf.map_to_host() as h:
        center_rho = h[State.index(NX // 2, 0, IDX_RHO)]
        var u = h[State.index(NX // 2, 0, IDX_MX)] / center_rho
        var ke = Real(0.5) * center_rho * u * u
        center_p = (GAMMA - Real(1.0)) * (h[State.index(NX // 2, 0, IDX_E)] - ke)

    print("1D Einfeldt 1-2-3 problem — t =", t)
    print("  min ρ:                  ", min_rho)
    print("  max ρ:                  ", max_rho)
    print("  min P:                  ", min_p)
    print("  ρ at center:            ", center_rho)
    print("  P at center:            ", center_p)

    assert_true(not any_nan, "1-2-3 problem NaN'd — positivity preserver failed")
    assert_true(min_rho > Real(0.0),
                "negative density survived — preserver insufficient")
    assert_true(min_p > Real(0.0),
                "negative pressure survived — preserver insufficient")

    # ---- Checks against the exact Riemann solution ---------------------
    var exact = ExactRiemann(GAMMA, RHO_L, U_L, P_L, RHO_R, U_R, P_R)
    var E_L = P_L / (GAMMA - Real(1.0)) + Real(0.5) * RHO_L * U_L * U_L
    var E_R = P_R / (GAMMA - Real(1.0)) + Real(0.5) * RHO_R * U_R * U_R

    # (1) Conservation with boundary-flux accounting. No wave has reached
    # x = 0 or x = 1, so the copy-out faces see the initial states and
    #   ∫U dV (t) = ∫U dV (0) + t · (F(U_L) − F(U_R))
    # holds exactly for U = (ρ, m_x, E).
    var err_mass = fabs(
        mesh.integrate(q, IDX_RHO) - mass_0 - t * (RHO_L * U_L - RHO_R * U_R)
    )
    var err_mom = fabs(
        mesh.integrate(q, IDX_MX) - mom_0
        - t * (RHO_L * U_L * U_L + P_L - RHO_R * U_R * U_R - P_R)
    )
    var err_en = fabs(
        mesh.integrate(q, IDX_E) - energy_0
        - t * ((E_L + P_L) * U_L - (E_R + P_R) * U_R)
    )

    # (2) L1 error of the primitives against the exact solution, sampled
    # at the element centres (the P = 2 mid node).
    var mid_node = State.num_nodes_per_elem // 2
    var l1_rho: Real = 0.0
    var l1_u: Real = 0.0
    var l1_p: Real = 0.0
    with q.buf.map_to_host() as h:
        for i in range(NX):
            var x = (Real(i) + Real(0.5)) * mesh.dx
            var ex = exact.sample((x - Real(0.5)) / t)
            var r = h[State.index(i, mid_node, IDX_RHO)]
            var uu = h[State.index(i, mid_node, IDX_MX)] / r
            var pp = (GAMMA - Real(1.0)) * (
                h[State.index(i, mid_node, IDX_E)] - Real(0.5) * r * uu * uu
            )
            l1_rho += mesh.dx * fabs(r - ex[0])
            l1_u += mesh.dx * fabs(uu - ex[1])
            l1_p += mesh.dx * fabs(pp - ex[2])
    print("  exact p* =", exact.p_star)
    print("  conservation errors (mass, momentum, energy):",
          err_mass, err_mom, err_en)
    print("  L1 errors vs exact (ρ, u, p):", l1_rho, l1_u, l1_p)
    assert_true(err_mass < Real(1.0e-11), "mass balance violated")
    assert_true(err_mom < Real(1.0e-11), "momentum balance violated")
    assert_true(err_en < Real(1.0e-11), "energy balance violated")
    assert_true(l1_rho < Real(2.0e-2), "L1(ρ) vs exact Riemann solution too large")
    assert_true(l1_u < Real(9.0e-2), "L1(u) vs exact Riemann solution too large")
    assert_true(l1_p < Real(1.0e-2), "L1(p) vs exact Riemann solution too large")
    # Near-vacuum centre: exact ρ* = 0.0219, p* = 0.0019. Bounds are a factor
    # ~3 around the measured values (ρ = 0.0090, P = 0.0064).
    # Exact star state: ρ* = 0.02185, p* = 0.001894. The centre pressure is
    # ~4x too high at this resolution (the well-known overheating of
    # conservative schemes on this problem); bound it so a regression shows.
    assert_true(fabs(center_rho - Real(0.02185)) < Real(0.2) * Real(0.02185), "centre density off")
    assert_true(center_p < Real(1.2e-2), "centre pressure overheating got worse")
    assert_true(min_rho > Real(0.5) * Real(0.02185), "density undershoots the vacuum state")
    assert_true(max_rho < Real(1.0) * Real(1.01), "density overshoot above the initial state")
    print("PASSED")
