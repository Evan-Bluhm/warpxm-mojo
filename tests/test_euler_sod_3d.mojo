# 3D Sod shock tube — 1D physics extruded along y and z.
#
# Initial discontinuity at x = 0.5; uniform in y and z. Sides perpendicular
# to x are copyout outflows; top/bottom and down/up are free-slip walls.
# Validates the 3D non-periodic BC plumbing on a problem with a known
# analytic structure.
#
# Expected outcome at t = 0.2:
#   * No NaN
#   * Result uniform in y and z at every x
#   * x-profile matches the 1D Sod analytic solution (within shock-capturing
#     tolerance) at four sample locations, and in L1 for (ρ, u, P) against
#     the exact Riemann solution
#   * Mass, x-momentum and energy obey the exact boundary-flux balance

from std.collections import Array
from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *
from warpxm.testing.riemann import ExactRiemann


comptime P: Int = 2
comptime NX: Int = 100
comptime NY: Int = 2
comptime NZ: Int = 2
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime DT: Real = 5.0e-4
comptime State = DGState[3, P, EULER_NC]

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
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1), Real(0.0), Real(0.1),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=False, periodic_z=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
        bc_down=BC_WALL_FREESLIP, bc_up=BC_WALL_FREESLIP,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, EULER_NC]:
        var rho = RHO_L if x < Real(0.5) else RHO_R
        var pp = P_L if x < Real(0.5) else P_R
        return [rho, Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)
    var mom_0 = mesh.integrate(q, IDX_MX)
    var energy_0 = mesh.integrate(q, IDX_E)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[3, P](app)
    var stepper = SSPRK3[3, P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    # Pull the mid-row (j=0, k=0) midpoint sample at every x.
    var mid_node = State.num_nodes_per_elem // 2
    var any_nan = False
    var max_y_drift: Real = 0.0
    var max_z_drift: Real = 0.0
    var rho_row = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        # First pass: row 0 in z=0 plane.
        for i in range(NX):
            var rho_ji = h[State.index(i, mid_node, IDX_RHO)]
            if isnan(rho_ji): any_nan = True
            rho_row[i] = rho_ji
        # Compare every other (j, k) sample to the row-0 z-0 value at the same x.
        for k in range(NZ):
            for j in range(NY):
                if j == 0 and k == 0: continue
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    var d = fabs(h[State.index(e, mid_node, IDX_RHO)] - rho_row[i])
                    if j != 0 and d > max_y_drift: max_y_drift = d
                    if k != 0 and d > max_z_drift: max_z_drift = d

    assert_true(not any_nan, "3D Sod NaN'd")
    print("max y-drift in ρ:", max_y_drift)
    print("max z-drift in ρ:", max_z_drift)
    assert_true(max_y_drift < Real(1.0e-10), "result not uniform in y")
    assert_true(max_z_drift < Real(1.0e-10), "result not uniform in z")

    var ix01 = Int(0.05 / mesh.dx)
    var ix06 = Int(0.6 / mesh.dx)
    var ix075 = Int(0.75 / mesh.dx)
    var ix095 = Int(0.95 / mesh.dx)
    print("ρ at x≈0.05 (left state ≈ 1.0):  ", rho_row[ix01])
    print("ρ at x≈0.6 (post-rarefaction ≈ 0.4263):", rho_row[ix06])
    print("ρ at x≈0.75 (post-contact ≈ 0.2655):  ", rho_row[ix075])
    print("ρ at x≈0.95 (right state ≈ 0.125):     ", rho_row[ix095])

    assert_true(fabs(rho_row[ix01] - Real(1.0)) < Real(1.0e-8), "left state off")
    assert_true(fabs(rho_row[ix06] - Real(0.42632)) < Real(4.0e-3), "post-rarefaction off")
    assert_true(fabs(rho_row[ix075] - Real(0.26557)) < Real(1.2e-3), "post-contact off")
    assert_true(fabs(rho_row[ix095] - Real(0.125)) < Real(1.0e-8), "right state off")

    # ---- Checks against the exact Riemann solution ---------------------
    var exact = ExactRiemann(GAMMA, RHO_L, U_L, P_L, RHO_R, U_R, P_R)
    var E_L = P_L / (GAMMA - Real(1.0)) + Real(0.5) * RHO_L * U_L * U_L
    var E_R = P_R / (GAMMA - Real(1.0)) + Real(0.5) * RHO_R * U_R * U_R

    # (1) Conservation with boundary-flux accounting. No wave has reached
    # x = 0 or x = 1, so the copy-out faces see the initial states and
    #   ∫U dV (t) = ∫U dV (0) + area · t · (F(U_L) − F(U_R))
    # holds exactly for U = (ρ, m_x, E).
    var area = (mesh.y_max - mesh.y_min) * (mesh.z_max - mesh.z_min)
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

    # (2) L1 error of the primitives against the exact solution, sampled
    # at the element centres of the first x-row (the P = 2 mid node).
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
    assert_true(l1_rho < Real(2.7e-2), "L1(ρ) vs exact Riemann solution too large")
    assert_true(l1_u < Real(5.0e-2), "L1(u) vs exact Riemann solution too large")
    assert_true(l1_p < Real(2.3e-2), "L1(p) vs exact Riemann solution too large")
    print("PASSED")
