# 2D Sod shock tube — extrudes the 1D Sod problem along y.
#
# Initial discontinuity at x = 0.5; uniform in y. Top/bottom walls are
# free-slip (= periodic in this case since no y-flux ever develops); left
# and right are copyout outflows.
#
# Expected outcome at t = 0.2:
#   * Result is uniform in y at every x (within machine precision).
#   * Profile in x matches the 1D Sod analytic solution to the same
#     tolerance as the 1D test (plateau samples + L1 error of ρ, u, P
#     against the exact Riemann solution).
#   * Mass, x-momentum and energy obey the exact boundary-flux balance.

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
comptime NY: Int = 2
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime DT: Real = 2.0e-4
comptime State = DGState[2, P, EULER_NC]

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
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = State(ctx, mesh.num_elements)

    # IC: classic Sod step in x, uniform in y.
    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
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
    var spatial = DGEulerSolver[2, P](app)
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    # Element-centre samples (the P = 2 mid node) of every row.
    var mid_node = State.num_nodes_per_elem // 2
    var any_nan = False
    var rho_at = Array[Real, NX](fill=Real(0.0))   # row 0 sample
    var max_y_drift: Real = 0.0
    var exact = ExactRiemann(GAMMA, RHO_L, U_L, P_L, RHO_R, U_R, P_R)
    var l1_rho: Real = 0.0
    var l1_u: Real = 0.0
    var l1_p: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var rho_ji = h[State.index(j * NX + i, mid_node, IDX_RHO)]
                if isnan(rho_ji):
                    any_nan = True
                if j == 0:
                    rho_at[i] = rho_ji
                else:
                    # Compare every other row to row 0 at the same x.
                    var d = fabs(rho_ji - rho_at[i])
                    if d > max_y_drift: max_y_drift = d
        # L1 error of the primitives against the exact solution, sampled
        # at the element centres of the first x-row.
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

    assert_true(not any_nan, "2D Sod result contains NaN")
    print("max y-drift in ρ (should be ~0):", max_y_drift)
    assert_true(max_y_drift < Real(1.0e-10),
                "result not uniform in y — 1D-extruded structure broken")

    # Sample x-profile values: plateaus from 1D Sod analytic.
    var ix01 = Int(0.1 / mesh.dx)
    var ix06 = Int(0.6 / mesh.dx)
    var ix075 = Int(0.75 / mesh.dx)
    var ix095 = Int(0.95 / mesh.dx)
    print("rho at x=0.1   :", rho_at[ix01], "  (analytic 1.0)")
    print("rho at x=0.6   :", rho_at[ix06], "  (analytic 0.4263)")
    print("rho at x=0.75  :", rho_at[ix075], "  (analytic 0.2655)")
    print("rho at x=0.95  :", rho_at[ix095], "  (analytic 0.125)")
    assert_true(fabs(rho_at[ix01] - Real(1.0)) < Real(1.0e-8), "left state off")
    assert_true(fabs(rho_at[ix06] - Real(0.42632)) < Real(8.0e-3), "post-rarefaction off")
    assert_true(fabs(rho_at[ix075] - Real(0.26557)) < Real(1.0e-3), "post-contact off")
    assert_true(fabs(rho_at[ix095] - Real(0.125)) < Real(1.0e-8), "right state off")

    # Conservation with boundary-flux accounting. No wave has reached
    # x = 0 or x = 1, so the copy-out faces see the initial states and
    #   ∫U dV (t) = ∫U dV (0) + area · t · (F(U_L) − F(U_R))
    # holds exactly for U = (ρ, m_x, E).
    var area = mesh.y_max - mesh.y_min
    var E_L = P_L / (GAMMA - Real(1.0)) + Real(0.5) * RHO_L * U_L * U_L
    var E_R = P_R / (GAMMA - Real(1.0)) + Real(0.5) * RHO_R * U_R * U_R
    var err_mass = fabs(
        mesh.integrate(q, IDX_RHO) - mass_0
        - area * t * (RHO_L * U_L - RHO_R * U_R)
    ) / area
    var err_mom = fabs(
        mesh.integrate(q, IDX_MX) - mom_0
        - area * t * (RHO_L * U_L * U_L + P_L - RHO_R * U_R * U_R - P_R)
    ) / area
    var err_en = fabs(
        mesh.integrate(q, IDX_E) - energy_0
        - area * t * ((E_L + P_L) * U_L - (E_R + P_R) * U_R)
    ) / area
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
