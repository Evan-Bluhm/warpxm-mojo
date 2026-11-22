# Brio-Wu MHD shock tube with the HLLD numerical flux (Miyoshi-Kusano).
# Mirrors `test_imhd_briowu_2d.mojo` exactly except the spatial solver is
# instantiated with `USE_HLLD=True`.  Validates that the HLLD path
# survives the full t = 0.1 run with the same correctness criteria
# (conservation / flux balance, constant B_x, isentropic fast
# rarefactions, continuity across the contact, contact sharpness).

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 2
comptime GAMMA: Real = 2.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 1.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=True,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var rho: Real
                    var pp: Real
                    var By: Real
                    if x < Real(0.5):
                        rho = 1.0; pp = 1.0; By = 1.0
                    else:
                        rho = 0.125; pp = 0.1; By = -1.0
                    var Bx_v: Real = 0.75
                    var bsq = Bx_v * Bx_v + By * By
                    var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
                    host[b + IDX_BX] = Bx_v
                    host[b + IDX_BY] = By
                    host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
    var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_NC](ctx, mesh.num_elements)


    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState2D[P, MHD_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var max_y_drift: Real = 0.0
    var rho_row0 = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                var b = (e * nodes_per_elem + mid_node) * MHD_NC
                if isnan(h[b + IDX_RHO]) or isnan(h[b + IDX_E]):
                    any_nan = True
                if j == 0:
                    rho_row0[i] = h[b + IDX_RHO]
                else:
                    var d = fabs(h[b + IDX_RHO] - rho_row0[i])
                    if d > max_y_drift: max_y_drift = d

    assert_true(not any_nan, "HLLD Brio-Wu produced NaN")
    print("HLLD Brio-Wu — t =", t)
    print("  max y-drift in ρ:   ", max_y_drift)
    assert_true(max_y_drift < Real(1.0e-10),
                "result should be uniform in y for a 1D-extruded Brio-Wu")
    var ix01 = Int(0.05 / mesh.dx)
    var ix099 = Int(0.95 / mesh.dx)
    print("  ρ at x≈0.05:        ", rho_row0[ix01])
    print("  ρ at x≈0.95:        ", rho_row0[ix099])
    assert_true(fabs(rho_row0[ix01] - Real(1.0)) < Real(2.0e-2),
                "left far-field density off")
    assert_true(fabs(rho_row0[ix099] - Real(0.125)) < Real(2.0e-2),
                "right far-field density off")

    # ---- Quantitative checks ------------------------------------------
    # All of these follow from the ideal-MHD equations for this Riemann
    # problem; none is a recorded "golden" number.
    var weights = NodalBasis1D[P].weights()
    var area = Real(2 * NY) * mesh.jacobian_y               # transverse cross-section
    var jac = mesh.jacobian_x * mesh.jacobian_y
    var int_rho: Real = 0.0
    var int_mx: Real = 0.0
    var int_my: Real = 0.0
    var int_E: Real = 0.0
    var int_By: Real = 0.0
    var int0_rho: Real = 0.0        # initial integrals, from the IC buffer
    var int0_E: Real = 0.0
    var int0_By: Real = 0.0
    var max_dBx: Real = 0.0
    var max_z: Real = 0.0           # |m_z| and |B_z| must stay 0
    var u_row = Array[Real, NX](fill=Real(0.0))
    var p_row = Array[Real, NX](fill=Real(0.0))
    var by_row = Array[Real, NX](fill=Real(0.0))
    var s_row = Array[Real, NX](fill=Real(0.0))    # entropy p / ρ^γ, γ = 2
    var rho_x = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            for n in range(nodes_per_elem):
                var w = jac
                var rem = n
                for _d in range(2):
                    w *= weights[rem % P1]
                    rem = rem // P1
                var b = (e * nodes_per_elem + n) * MHD_NC
                int_rho += w * h[b + IDX_RHO]
                int_mx += w * h[b + IDX_MX]
                int_my += w * h[b + IDX_MY]
                int_E += w * h[b + IDX_E]
                int_By += w * h[b + IDX_BY]
                int0_rho += w * host[b + IDX_RHO]
                int0_E += w * host[b + IDX_E]
                int0_By += w * host[b + IDX_BY]
                var dbx = fabs(h[b + IDX_BX] - Real(0.75))
                if dbx > max_dBx: max_dBx = dbx
                if fabs(h[b + IDX_MZ]) > max_z: max_z = fabs(h[b + IDX_MZ])
                if fabs(h[b + IDX_BZ]) > max_z: max_z = fabs(h[b + IDX_BZ])
        # x-profile of primitives at the element mid-nodes of the first row.
        for i in range(NX):
            var b = (i * nodes_per_elem + nodes_per_elem // 2) * MHD_NC
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
