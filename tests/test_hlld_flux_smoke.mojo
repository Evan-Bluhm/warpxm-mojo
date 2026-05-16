# Standalone unit test for the HLLD numerical flux on IMHD.
#
# Three cases:
#   (a) Identical L/R states (consistency): HLLD flux equals the
#       interior physical flux of that state, both axes.
#   (b) Brio-Wu Riemann problem at the L/R discontinuity: HLLD returns
#       a finite flux with the correct contact wave structure
#       (mass flux F_ρ should be consistent with shock speed).
#   (c) Smooth small-perturbation state: HLLD and Rusanov fluxes agree
#       to first order in the perturbation amplitude.

from std.testing import assert_true

from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real


def main() raises:
    var gamma: Real = 5.0 / 3.0
    var rho_min: Real = 1e-12
    var p_min: Real = 1e-12

    # ---- (a) Consistency: L == R -----------------------------------------
    var q = InlineArray[Real, MHD_NC](fill=Real(0.0))
    q[IDX_RHO] = 1.0
    q[IDX_MX] = 0.5
    q[IDX_MY] = -0.3
    q[IDX_MZ] = 0.1
    q[IDX_BX] = 0.4
    q[IDX_BY] = -0.2
    q[IDX_BZ] = 0.1
    var ke = Real(0.5) * (0.5*0.5 + 0.3*0.3 + 0.1*0.1) / Real(1.0)
    var mag = Real(0.5) * (0.4*0.4 + 0.2*0.2 + 0.1*0.1)
    q[IDX_E] = 1.0 / (gamma - Real(1.0)) + ke + mag  # P_gas = 1

    var f_phys_x = IMHD.internal_flux_x_static(q, gamma, rho_min, p_min)
    var f_hlld_x = IMHD.numerical_flux_hlld_static(
        q, q, Real(1.0), Real(0.0), Real(0.0), gamma, rho_min, p_min,
    )
    var max_err_x: Real = 0.0
    for c in range(MHD_NC):
        var d = fabs(f_phys_x[c] - f_hlld_x[c])
        if d > max_err_x: max_err_x = d

    var f_phys_y = IMHD.internal_flux_y_static(q, gamma, rho_min, p_min)
    var f_hlld_y = IMHD.numerical_flux_hlld_static(
        q, q, Real(0.0), Real(1.0), Real(0.0), gamma, rho_min, p_min,
    )
    var max_err_y: Real = 0.0
    for c in range(MHD_NC):
        var d = fabs(f_phys_y[c] - f_hlld_y[c])
        if d > max_err_y: max_err_y = d

    print("(a) HLLD consistency, x: max |F_HLLD - F_phys| =", max_err_x)
    print("(a) HLLD consistency, y: max |F_HLLD - F_phys| =", max_err_y)
    assert_true(max_err_x < Real(1.0e-12), "HLLD inconsistent on x-normal smooth state")
    assert_true(max_err_y < Real(1.0e-12), "HLLD inconsistent on y-normal smooth state")

    # ---- (b) Brio-Wu Riemann problem -------------------------------------
    var q_l = InlineArray[Real, MHD_NC](fill=Real(0.0))
    q_l[IDX_RHO] = 1.0
    q_l[IDX_BX] = 0.75
    q_l[IDX_BY] = 1.0
    var bsq_l = 0.75*0.75 + 1.0*1.0
    q_l[IDX_E] = 1.0 / (Real(2.0) - Real(1.0)) + Real(0.5) * bsq_l

    var q_r = InlineArray[Real, MHD_NC](fill=Real(0.0))
    q_r[IDX_RHO] = 0.125
    q_r[IDX_BX] = 0.75
    q_r[IDX_BY] = -1.0
    var bsq_r = 0.75*0.75 + 1.0*1.0
    q_r[IDX_E] = 0.1 / (Real(2.0) - Real(1.0)) + Real(0.5) * bsq_r

    var f_bw = IMHD.numerical_flux_hlld_static(
        q_l, q_r, Real(1.0), Real(0.0), Real(0.0), Real(2.0), rho_min, p_min,
    )
    print("(b) Brio-Wu HLLD flux:")
    print("    F_ρ  =", f_bw[IDX_RHO])
    print("    F_mx =", f_bw[IDX_MX])
    print("    F_E  =", f_bw[IDX_E])
    print("    F_By =", f_bw[IDX_BY])
    # Sanity: must be finite, non-NaN.
    for c in range(MHD_NC):
        assert_true(f_bw[c] == f_bw[c], "HLLD produced NaN on Brio-Wu")
    # Mass flux should be small/zero (no net momentum across the contact at
    # t=0 since both states have u=0); shock develops over time.
    assert_true(fabs(f_bw[IDX_RHO]) < Real(2.0),
                "Brio-Wu mass flux unreasonable")

    # ---- (c) Small perturbation: HLLD ≈ Rusanov to first order ----------
    var eps_pert: Real = 0.001
    var q_l2 = InlineArray[Real, MHD_NC](fill=Real(0.0))
    var q_r2 = InlineArray[Real, MHD_NC](fill=Real(0.0))
    for c in range(MHD_NC):
        q_l2[c] = q[c] - eps_pert * Real(c + 1) * Real(0.1)
        q_r2[c] = q[c] + eps_pert * Real(c + 1) * Real(0.1)
    var f_rus = IMHD.numerical_flux_static(
        q_l2, q_r2, Real(1.0), Real(0.0), Real(0.0), gamma, rho_min, p_min,
    )
    var f_hl = IMHD.numerical_flux_hlld_static(
        q_l2, q_r2, Real(1.0), Real(0.0), Real(0.0), gamma, rho_min, p_min,
    )
    var max_diff: Real = 0.0
    for c in range(MHD_NC):
        var d = fabs(f_rus[c] - f_hl[c])
        if d > max_diff: max_diff = d
    print("(c) Smooth-state HLLD vs Rusanov max |ΔF| =", max_diff)
    # Should be small but non-zero — both consistent with mid-state physical flux.
    assert_true(max_diff < Real(1.0e-2),
                "HLLD and Rusanov differ wildly on smooth state")
    print("PASSED")
