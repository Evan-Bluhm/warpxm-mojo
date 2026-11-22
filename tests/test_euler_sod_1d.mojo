# Regression test: 1D Sod shock tube with the Moe-Rossmanith limiter.
#
# Asserts that the limited DG scheme:
#   * does NOT NaN out (positivity preserved);
#   * places the shock, contact, and rarefaction at the analytic Riemann
#     locations to within one element of accuracy at t = 0.2.
#
# Analytic at t = 0.2 (γ = 1.4, x_0 = 0.5, t = 0.2):
#   Rarefaction head at x ≈ 0.264, tail at x ≈ 0.486
#   Post-rarefaction plateau (left of contact):
#     ρ_3 ≈ 0.4263, u_3 ≈ 0.9275, P_3 ≈ 0.3031
#   Contact at x ≈ 0.685
#   Post-contact plateau (right of contact):
#     ρ_4 ≈ 0.2655, u_4 ≈ 0.9275, P_4 ≈ 0.3031
#   Shock at x ≈ 0.850.
#
# Checks: plateau values (ρ, u, P) at four sample locations, exact
# conservation of mass / momentum / energy with boundary-flux accounting,
# and the L1 error of (ρ, u, P) against the exact Riemann solution
# (`warpxm.testing.riemann.ExactRiemann`).

from std.collections import Array
from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *
from warpxm.testing.riemann import ExactRiemann


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 200
comptime GAMMA: Real = 1.4
comptime State = DGState[1, P, EULER_NC]

# Left / right primitive states (ρ, u, p).
comptime RHO_L: Real = 1.0
comptime U_L: Real = 0.0
comptime P_L: Real = 1.0
comptime RHO_R: Real = 0.125
comptime U_R: Real = 0.0
comptime P_R: Real = 0.1


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, EULER_NC]:
        var rho = RHO_L if x < Real(0.5) else RHO_R
        var pp = P_L if x < Real(0.5) else P_R
        return [rho, Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)
    var mom_0 = mesh.integrate(q, IDX_MX)
    var energy_0 = mesh.integrate(q, IDX_E)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[1, P](app)
    var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[1, P, EULER_NC](ctx, mesh.num_elements)

    var dt: Real = 2.0e-4
    var t_final: Real = 0.2

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, adjust,
    )
    ctx.synchronize()

    # Sample at element midpoints; assert (a) finiteness, (b) plateau / right
    # state values, (c) shock and contact in roughly the right locations.
    var mid_node = State.num_nodes_per_elem // 2
    var any_nan = False
    var rho_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    var u_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    var p_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var rho = h[State.index(e, mid_node, IDX_RHO)]
            var mx = h[State.index(e, mid_node, IDX_MX)]
            var e_t = h[State.index(e, mid_node, IDX_E)]
            if isnan(rho) or isnan(mx) or isnan(e_t):
                any_nan = True
            rho_at_x[e] = rho
            u_at_x[e] = mx / rho
            var ke = Real(0.5) * rho * u_at_x[e] * u_at_x[e]
            p_at_x[e] = (GAMMA - Real(1.0)) * (e_t - ke)
    assert_true(not any_nan, "Sod result contains NaN")

    # x = 0.1 → left undisturbed state (ρ = 1, u = 0, P = 1).
    var ix_left = Int(0.1 / mesh.dx)
    print("at x=0.1:  rho =", rho_at_x[ix_left],
          "u =", u_at_x[ix_left], "P =", p_at_x[ix_left])
    assert_true(fabs(rho_at_x[ix_left] - Real(1.0)) < Real(1.0e-8),
                "left undisturbed state ρ wrong")

    # x = 0.6 → post-rarefaction plateau.
    var ix_plateau1 = Int(0.6 / mesh.dx)
    print("at x=0.6:  rho =", rho_at_x[ix_plateau1],
          "u =", u_at_x[ix_plateau1], "P =", p_at_x[ix_plateau1])
    assert_true(fabs(rho_at_x[ix_plateau1] - Real(0.42632)) < Real(8.0e-3),
                "post-rarefaction ρ outside tolerance")
    assert_true(fabs(u_at_x[ix_plateau1] - Real(0.92745)) < Real(3.0e-3),
                "post-rarefaction u outside tolerance")
    assert_true(fabs(p_at_x[ix_plateau1] - Real(0.30313)) < Real(1.0e-3),
                "post-rarefaction P outside tolerance")

    # x = 0.75 → post-contact plateau.
    var ix_plateau2 = Int(0.75 / mesh.dx)
    print("at x=0.75: rho =", rho_at_x[ix_plateau2],
          "u =", u_at_x[ix_plateau2], "P =", p_at_x[ix_plateau2])
    assert_true(fabs(rho_at_x[ix_plateau2] - Real(0.26557)) < Real(1.0e-3),
                "post-contact ρ outside tolerance")
    # u and P are continuous across the contact.
    assert_true(fabs(u_at_x[ix_plateau2] - Real(0.92745)) < Real(3.0e-3),
                "post-contact u outside tolerance")
    assert_true(fabs(p_at_x[ix_plateau2] - Real(0.30313)) < Real(1.0e-3),
                "post-contact P outside tolerance")

    # x = 0.95 → right undisturbed state.
    var ix_right = Int(0.95 / mesh.dx)
    print("at x=0.95: rho =", rho_at_x[ix_right],
          "u =", u_at_x[ix_right], "P =", p_at_x[ix_right])
    assert_true(fabs(rho_at_x[ix_right] - Real(0.125)) < Real(1.0e-8),
                "right undisturbed state ρ wrong")

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
    var l1_rho: Real = 0.0
    var l1_u: Real = 0.0
    var l1_p: Real = 0.0
    for i in range(NUM_ELEMENTS):
        var x = (Real(i) + Real(0.5)) * mesh.dx
        var ex = exact.sample((x - Real(0.5)) / t)
        l1_rho += mesh.dx * fabs(rho_at_x[i] - ex[0])
        l1_u += mesh.dx * fabs(u_at_x[i] - ex[1])
        l1_p += mesh.dx * fabs(p_at_x[i] - ex[2])
    print("  exact p* =", exact.p_star)
    print("  conservation errors (mass, momentum, energy):",
          err_mass, err_mom, err_en)
    print("  L1 errors vs exact (ρ, u, p):", l1_rho, l1_u, l1_p)
    assert_true(err_mass < Real(1.0e-11), "mass balance violated")
    assert_true(err_mom < Real(1.0e-11), "momentum balance violated")
    assert_true(err_en < Real(1.0e-11), "energy balance violated")
    assert_true(l1_rho < Real(1.5e-2), "L1(ρ) vs exact Riemann solution too large")
    assert_true(l1_u < Real(3.0e-2), "L1(u) vs exact Riemann solution too large")
    assert_true(l1_p < Real(1.3e-2), "L1(p) vs exact Riemann solution too large")
    print("PASSED")
