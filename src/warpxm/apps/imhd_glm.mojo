# Ideal MHD with Dedner-style MixedGLM divergence cleaning — single
# dimension-agnostic app.
#
# Reference: Dedner, Kemm, Kröner, Munz, Schnitzer & Wesenberg, "Hyperbolic
# Divergence Cleaning for the MHD Equations", J. Comp. Phys. 175, 645 (2002).
#
# 9 conserved variables: q = (ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z, ψ).
# The standard ideal-MHD system preserves ∇·B = 0 analytically, but
# numerical schemes accumulate ∇·B errors at shocks. MixedGLM augments
# the system with a 9th conserved scalar ψ:
#
#     ∂B/∂t + ∇·(uB − Bu) + ∇ψ      = 0
#     ∂ψ/∂t + c_h² · ∇·B           = −(c_h² / c_p²) · ψ
#
# `c_h` is the hyperbolic cleaning speed; `c_p²` controls parabolic
# damping (smaller ⇒ stronger damping).
#
# Per-axis fluxes add the ψ-coupling on top of the standard IMHD flux:
#
#     f_x[B_x] = ψ        f_y[B_y] = ψ        f_z[B_z] = ψ
#     f_x[ψ]   = c_h²·B_x  f_y[ψ]   = c_h²·B_y  f_z[ψ]   = c_h²·B_z
#
# Numerical flux options:
#   * Rusanov: c_max = fmax(MHD c_max, c_h).
#   * HLLD: 8 MHD components via IMHD HLLD; (B_n, ψ) block solved exactly
#     as a decoupled 2x2 Riemann at speeds ±c_h.

from std.collections import Array

from warpxm.apps.imhd import (
    IMHD, MHD_NC,
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
    _normal_flux_and_speed,
)
from warpxm.core.math_compat import fabs, fmax
from warpxm.core.types import Real


comptime MHD_GLM_NC: Int = 9
comptime IDX_PSI: Int = 8


@fieldwise_init
struct ImhdGLM(ImplicitlyCopyable, Movable):
    """Ideal MHD + Dedner MixedGLM divergence cleaning."""

    var gas_gamma: Real
    var min_density: Real
    var min_pressure: Real
    var cfl_max: Real
    var c_h: Real
    """Hyperbolic cleaning speed (usually the max signal speed on the mesh)."""

    var c_p_sq: Real
    """Parabolic damping coefficient. Smaller ⇒ faster ψ decay.
    Common choice: `c_p² = 0.18 · c_h · L_domain` (Mignone & Tzeferacos 2010).
    """

    @staticmethod
    def num_components() -> Int:
        return MHD_GLM_NC

    # ------------------------------------------------------------------
    # Internal flux: reuse the 8-component IMHD flux for the MHD piece
    # and add the ψ-coupling on top.
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_x_static(
        q: Array[Real, MHD_GLM_NC],
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> Array[Real, MHD_GLM_NC]:
        var q_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            q_mhd[c] = q[c]
        var f_mhd = IMHD.internal_flux_x_static(q_mhd, gamma, rho_min, press_min)
        var f: Array[Real, MHD_GLM_NC] = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            f[c] = f_mhd[c]
        f[IDX_BX] = q[IDX_PSI]
        f[IDX_PSI] = c_h * c_h * q[IDX_BX]
        return f^

    @staticmethod
    def internal_flux_y_static(
        q: Array[Real, MHD_GLM_NC],
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> Array[Real, MHD_GLM_NC]:
        var q_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            q_mhd[c] = q[c]
        var f_mhd = IMHD.internal_flux_y_static(q_mhd, gamma, rho_min, press_min)
        var f: Array[Real, MHD_GLM_NC] = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            f[c] = f_mhd[c]
        f[IDX_BY] = q[IDX_PSI]
        f[IDX_PSI] = c_h * c_h * q[IDX_BY]
        return f^

    @staticmethod
    def internal_flux_z_static(
        q: Array[Real, MHD_GLM_NC],
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> Array[Real, MHD_GLM_NC]:
        var q_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            q_mhd[c] = q[c]
        var f_mhd = IMHD.internal_flux_z_static(q_mhd, gamma, rho_min, press_min)
        var f: Array[Real, MHD_GLM_NC] = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            f[c] = f_mhd[c]
        f[IDX_BZ] = q[IDX_PSI]
        f[IDX_PSI] = c_h * c_h * q[IDX_BZ]
        return f^

    # ------------------------------------------------------------------
    # Rusanov numerical flux on a 3-component unit normal. Max signal
    # speed is the larger of the IMHD fast magnetosonic speed and c_h.
    # ------------------------------------------------------------------

    @staticmethod
    @always_inline
    def max_signal_speed_static(
        q: Array[Real, MHD_GLM_NC],
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> Real:
        """The ideal-MHD bound or the cleaning speed `c_h`, whichever is
        larger. For CFL time-step selection."""
        var q_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        comptime for k in range(MHD_NC):
            q_mhd[k] = q[k]
        return fmax(
            IMHD.max_signal_speed_static(q_mhd, gamma, rho_min, press_min), c_h,
        )

    @staticmethod
    def numerical_flux_static(
        q_l: Array[Real, MHD_GLM_NC],
        q_r: Array[Real, MHD_GLM_NC],
        nx: Real, ny: Real, nz: Real,
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> Array[Real, MHD_GLM_NC]:
        # The 8 MHD components come from the shared one-evaluation-per-side
        # normal flux; the (B·n, ψ) coupling is added on top.
        var q_l_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        var q_r_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            q_l_mhd[c] = q_l[c]
            q_r_mhd[c] = q_r[c]
        var fs_l = _normal_flux_and_speed(q_l_mhd, nx, ny, nz, gamma, rho_min, press_min)
        var fs_r = _normal_flux_and_speed(q_r_mhd, nx, ny, nz, gamma, rho_min, press_min)
        var c_max = fmax(fmax(fs_l[MHD_NC], fs_r[MHD_NC]), c_h)

        var B_n_l = q_l[IDX_BX] * nx + q_l[IDX_BY] * ny + q_l[IDX_BZ] * nz
        var B_n_r = q_r[IDX_BX] * nx + q_r[IDX_BY] * ny + q_r[IDX_BZ] * nz
        var f_n_l: Array[Real, MHD_GLM_NC] = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        var f_n_r: Array[Real, MHD_GLM_NC] = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            f_n_l[c] = fs_l[c]
            f_n_r[c] = fs_r[c]
        # Internal flux has f_x[B_x] = ψ (replacing the MHD zero) and
        # f_x[ψ] = c_h² B_x, so F·n[B_i] gains ψ n_i and F·n[ψ] = c_h² B·n.
        f_n_l[IDX_BX] += q_l[IDX_PSI] * nx
        f_n_l[IDX_BY] += q_l[IDX_PSI] * ny
        f_n_l[IDX_BZ] += q_l[IDX_PSI] * nz
        f_n_r[IDX_BX] += q_r[IDX_PSI] * nx
        f_n_r[IDX_BY] += q_r[IDX_PSI] * ny
        f_n_r[IDX_BZ] += q_r[IDX_PSI] * nz
        f_n_l[IDX_PSI] = c_h * c_h * B_n_l
        f_n_r[IDX_PSI] = c_h * c_h * B_n_r

        var f: Array[Real, MHD_GLM_NC] = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_GLM_NC):
            f[c] = Real(0.5) * (f_n_l[c] + f_n_r[c]) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f^

    # ------------------------------------------------------------------
    # HLLD with GLM divergence cleaning. The (B·n, ψ) subsystem is a 2x2
    # linear hyperbolic block decoupled from the standard MHD waves
    # (Dedner et al. 2002 §3.2; Mignone & Tzeferacos 2010 §4):
    #     ψ̃   = ½(ψ_L + ψ_R) − ½·c_h·(B_n_R − B_n_L)
    #     B̃_n = ½(B_n_L + B_n_R) − (ψ_R − ψ_L)/(2·c_h)
    # giving fluxes F[B_n] += ψ̃ and F[ψ] = c_h²·B̃_n. The 8 MHD
    # components come from IMHD HLLD.
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_hlld_static(
        q_l: Array[Real, MHD_GLM_NC],
        q_r: Array[Real, MHD_GLM_NC],
        nx: Real, ny: Real, nz: Real,
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> Array[Real, MHD_GLM_NC]:
        var q_l_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        var q_r_mhd: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            q_l_mhd[c] = q_l[c]
            q_r_mhd[c] = q_r[c]
        var f_mhd = IMHD.numerical_flux_hlld_static(
            q_l_mhd, q_r_mhd, nx, ny, nz, gamma, rho_min, press_min,
        )

        var B_n_l = q_l[IDX_BX] * nx + q_l[IDX_BY] * ny + q_l[IDX_BZ] * nz
        var B_n_r = q_r[IDX_BX] * nx + q_r[IDX_BY] * ny + q_r[IDX_BZ] * nz
        var psi_l = q_l[IDX_PSI]
        var psi_r = q_r[IDX_PSI]
        var psi_hat = Real(0.5) * (psi_l + psi_r) - Real(0.5) * c_h * (B_n_r - B_n_l)
        var B_n_hat = Real(0.5) * (B_n_l + B_n_r) - Real(0.5) * (psi_r - psi_l) / c_h

        var f: Array[Real, MHD_GLM_NC] = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            f[c] = f_mhd[c]
        f[IDX_BX] += nx * psi_hat
        f[IDX_BY] += ny * psi_hat
        f[IDX_BZ] += nz * psi_hat
        f[IDX_PSI] = c_h * c_h * B_n_hat
        return f^
