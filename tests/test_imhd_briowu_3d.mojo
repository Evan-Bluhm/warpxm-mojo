# 3D Brio-Wu MHD shock tube — 1D physics extruded along y and z.
#
# Same Riemann problem as `test_imhd_briowu_2d.mojo` lifted into 3D: an
# x-discontinuity at x = 0.5, periodic in y, periodic in z (free choice
# since the IC is uniform in those axes), copyout on x. γ = 2.
#
# Left state  (x < 0.5):  ρ = 1,     P = 1,    B_y = +1
# Right state (x ≥ 0.5):  ρ = 0.125, P = 0.1,  B_y = -1
# Both states:             u = 0, B_x = 0.75, B_z = 0.
#
# Validation:
#   * no NaN
#   * solution remains uniform in y and z (1D extruded)
#   * far-field x states preserved (shock hasn't reached them at t = 0.1)
#   * exact conservation of ρ, E, B_y and the exact boundary-flux balance of
#     m_x and m_y (copy-out faces still see the initial states)
#   * B_x stays 0.75 everywhere; m_z and B_z are never excited
#   * both fast rarefactions are isentropic (p/ρ^γ unchanged)
#   * u, p, B_y are continuous across the contact while ρ jumps
#   * the contact is captured within a few elements

from std.collections import Array
from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 2
comptime NZ: Int = 2
comptime GAMMA: Real = 2.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 1.0e-4
comptime State = DGState[3, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1), Real(0.0), Real(0.1),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=True, periodic_z=True,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, MHD_NC]:
        var left = x < Real(0.5)
        var rho: Real = 1.0 if left else 0.125
        var pp: Real = 1.0 if left else 0.1
        var by: Real = 1.0 if left else -1.0
        var bx: Real = 0.75
        var v = Array[Real, MHD_NC](fill=Real(0.0))
        v[IDX_RHO] = rho
        v[IDX_E] = pp / (GAMMA - Real(1.0)) + Real(0.5) * (bx * bx + by * by)
        v[IDX_BX] = bx
        v[IDX_BY] = by
        return v^
    mesh.fill(q, initial)
    var int0_rho = mesh.integrate(q, IDX_RHO)
    var int0_E = mesh.integrate(q, IDX_E)
    var int0_By = mesh.integrate(q, IDX_BY)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var mid_node = State.num_nodes_per_elem // 2
    var any_nan = False
    var max_y_drift: Real = 0.0
    var max_z_drift: Real = 0.0
    var rho_row = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        # First pass: row j=0, k=0.
        for i in range(NX):
            var e = 0 * NX * NY + 0 * NX + i
            var b = State.index(e, mid_node, 0)
            var rho_v = h[b + IDX_RHO]
            if isnan(rho_v): any_nan = True
            rho_row[i] = rho_v
        for k in range(NZ):
            for j in range(NY):
                if k == 0 and j == 0: continue
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    var b = State.index(e, mid_node, 0)
                    var d = fabs(h[b + IDX_RHO] - rho_row[i])
                    if j != 0 and d > max_y_drift: max_y_drift = d
                    if k != 0 and d > max_z_drift: max_z_drift = d

    assert_true(not any_nan, "3D Brio-Wu NaN'd")
    print("max y-drift in ρ (uniformity check):", max_y_drift)
    print("max z-drift in ρ:                   ", max_z_drift)
    assert_true(max_y_drift < Real(1.0e-10), "result not uniform in y")
    assert_true(max_z_drift < Real(1.0e-10), "result not uniform in z")

    # Far-field state sanity checks. At t = 0.1 the shock fronts haven't
    # propagated far enough to disturb x ≈ 0.05 or x ≈ 0.95.
    var ix005 = Int(0.05 / mesh.dx)
    var ix095 = Int(0.95 / mesh.dx)
    print("ρ at x≈0.05 (left state ≈ 1.0):  ", rho_row[ix005])
    print("ρ at x≈0.95 (right state ≈ 0.125):", rho_row[ix095])
    assert_true(fabs(rho_row[ix005] - Real(1.0)) < Real(2.0e-2), "left far-field off")
    assert_true(fabs(rho_row[ix095] - Real(0.125)) < Real(2.0e-2), "right far-field off")

    # ---- Quantitative checks ------------------------------------------
    # All of these follow from the ideal-MHD equations for this Riemann
    # problem; none is a recorded "golden" number.
    var area = Real(4 * NY * NZ) * mesh.jacobian_y * mesh.jacobian_z               # transverse cross-section
    var int_rho: Real = 0.0
    var int_mx: Real = 0.0
    var int_my: Real = 0.0
    var int_E: Real = 0.0
    var int_By: Real = 0.0
    var max_dBx: Real = 0.0
    var max_z: Real = 0.0           # |m_z| and |B_z| must stay 0
    var u_row = Array[Real, NX](fill=Real(0.0))
    var p_row = Array[Real, NX](fill=Real(0.0))
    var by_row = Array[Real, NX](fill=Real(0.0))
    var s_row = Array[Real, NX](fill=Real(0.0))    # entropy p / ρ^γ, γ = 2
    var rho_x = Array[Real, NX](fill=Real(0.0))
    def totals(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, MHD_NC],
    ) {mut int_rho, mut int_mx, mut int_my, mut int_E, mut int_By, mut max_dBx, mut max_z}:
        int_rho += w * v[IDX_RHO]
        int_mx += w * v[IDX_MX]
        int_my += w * v[IDX_MY]
        int_E += w * v[IDX_E]
        int_By += w * v[IDX_BY]
        var dbx = fabs(v[IDX_BX] - Real(0.75))
        if dbx > max_dBx: max_dBx = dbx
        if fabs(v[IDX_MZ]) > max_z: max_z = fabs(v[IDX_MZ])
        if fabs(v[IDX_BZ]) > max_z: max_z = fabs(v[IDX_BZ])
    mesh.visit_nodes(q, totals)
    with q.buf.map_to_host() as h:
        # x-profile of primitives at the element mid-nodes of the first row.
        for i in range(NX):
            var b = State.index(i, mid_node, 0)
            var r = h[b + IDX_RHO]
            var ke = Real(0.5) * (
                h[b + IDX_MX] * h[b + IDX_MX] + h[b + IDX_MY] * h[b + IDX_MY]
                + h[b + IDX_MZ] * h[b + IDX_MZ]
            ) / r
            var me = Real(0.5) * (
                h[b + IDX_BX] * h[b + IDX_BX] + h[b + IDX_BY] * h[b + IDX_BY]
                + h[b + IDX_BZ] * h[b + IDX_BZ]
            )
            var pp = (GAMMA - Real(1.0)) * (h[b + IDX_E] - ke - me)
            rho_x[i] = r
            u_row[i] = h[b + IDX_MX] / r
            p_row[i] = pp
            by_row[i] = h[b + IDX_BY]
            s_row[i] = pp / (r * r)

    # (1) Conservation with boundary-flux accounting. The fast waves have
    # not reached x = 0 or x = 1, so the copy-out faces see the initial
    # states and d/dt ∫U dV = area · (F_x(U_L) − F_x(U_R)) exactly:
    #   ρ, E, B_y : F_L = F_R (u = 0 on both sides)      → conserved
    #   m_x       : F = p + (B_y² − B_x²)/2 → F_L − F_R = p_L − p_R = 0.9
    #   m_y       : F = −B_x B_y           → F_L − F_R = −1.5
    var err_rho = fabs(int_rho - int0_rho)
    var err_E = fabs(int_E - int0_E)
    var err_By = fabs(int_By - int0_By)
    var err_mx = fabs(int_mx - area * Real(0.9) * t)
    var err_my = fabs(int_my + area * Real(1.5) * t)
    print("  conservation errors / area (ρ, m_x, m_y, E, B_y):",
          err_rho / area, err_mx / area, err_my / area, err_E / area,
          err_By / area)
    assert_true(err_rho < Real(1.0e-11) * area, "mass not conserved")
    assert_true(err_mx < Real(1.0e-11) * area, "x-momentum balance (p_L − p_R)·t violated")
    assert_true(err_my < Real(1.0e-11) * area, "y-momentum balance −2 B_x B_y·t violated")
    assert_true(err_E < Real(1.0e-11) * area, "energy not conserved")
    assert_true(err_By < Real(1.0e-11) * area, "∫B_y not conserved")

    # (2) B_x is constant in 1D (∇·B = 0); m_z and B_z are never excited.
    print("  max |B_x − 0.75|:", max_dBx, "  max |m_z|, |B_z|:", max_z)
    assert_true(max_dBx < Real(1.0e-10), "B_x drifted (∇·B error)")
    assert_true(max_z < Real(1.0e-10), "z components were excited")

    # (3) Both fast rarefactions are isentropic, so p/ρ^γ keeps its initial
    # value everywhere outside the slow waves: 1 for x < 0.43 (left of the
    # slow compound wave) and 0.1/0.125² = 6.4 for x > 0.67 (right of the
    # slow shock). A wrong flux or an over-dissipative limiter produces
    # entropy inside the fans.
    var max_ds_left: Real = 0.0
    var max_ds_right: Real = 0.0
    for i in range(NX):
        var x = (Real(i) + Real(0.5)) * mesh.dx
        if x < Real(0.43):
            var d = fabs(s_row[i] - Real(1.0))
            if d > max_ds_left: max_ds_left = d
        if x > Real(0.67):
            var d = fabs(s_row[i] / Real(6.4) - Real(1.0))
            if d > max_ds_right: max_ds_right = d
    print("  max relative entropy error in the fast rarefactions (L, R):",
          max_ds_left, max_ds_right)
    assert_true(max_ds_left < Real(1.0e-3), "left fast rarefaction not isentropic")
    assert_true(max_ds_right < Real(1.0e-3), "right fast rarefaction not isentropic")

    # (4) Across the contact (x = 0.5 + u·t ≈ 0.56) only ρ jumps: with
    # B_x ≠ 0, u, p and B_y are continuous. Compare plateau means on the two
    # sides (x ∈ [0.51, 0.555] vs [0.575, 0.635]).
    var p_a: Real = 0.0
    var u_a: Real = 0.0
    var by_a: Real = 0.0
    var rho_a: Real = 0.0
    var p_b: Real = 0.0
    var u_b: Real = 0.0
    var by_b: Real = 0.0
    var rho_b: Real = 0.0
    var n_a = 0
    var n_b = 0
    for i in range(NX):
        var x = (Real(i) + Real(0.5)) * mesh.dx
        if x > Real(0.51) and x < Real(0.555):
            p_a += p_row[i]; u_a += u_row[i]; by_a += by_row[i]; rho_a += rho_x[i]
            n_a += 1
        if x > Real(0.575) and x < Real(0.635):
            p_b += p_row[i]; u_b += u_row[i]; by_b += by_row[i]; rho_b += rho_x[i]
            n_b += 1
    p_a /= Real(n_a); u_a /= Real(n_a); by_a /= Real(n_a); rho_a /= Real(n_a)
    p_b /= Real(n_b); u_b /= Real(n_b); by_b /= Real(n_b); rho_b /= Real(n_b)
    print("  contact: left  (ρ, u, p, B_y) =", rho_a, u_a, p_a, by_a)
    print("           right (ρ, u, p, B_y) =", rho_b, u_b, p_b, by_b)
    assert_true(fabs(p_a - p_b) < Real(1.5e-2), "pressure jumps across the contact")
    assert_true(fabs(u_a - u_b) < Real(1.5e-2), "u jumps across the contact")
    assert_true(fabs(by_a - by_b) < Real(1.5e-2), "B_y jumps across the contact")
    assert_true(rho_a - rho_b > Real(0.4), "contact density jump missing")

    # (5) The contact is captured sharply: the largest element-to-element
    # density step is a sizeable fraction of the 0.46 jump.
    var max_step: Real = 0.0
    for i in range(NX - 1):
        var g = fabs(rho_x[i + 1] - rho_x[i])
        if g > max_step: max_step = g
    print("  max element-to-element ρ step:", max_step)
    assert_true(max_step > Real(0.12), "contact smeared over too many elements")
    print("PASSED")
