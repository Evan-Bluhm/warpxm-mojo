# Ideal MHD with Dedner-style MixedGLM divergence cleaning, 2D.
#
# Reference: Dedner, Kemm, Kröner, Munz, Schnitzer & Wesenberg, "Hyperbolic
# Divergence Cleaning for the MHD Equations", J. Comp. Phys. 175, 645 (2002).
#
# The standard ideal-MHD system preserves ∇·B = 0 analytically, but
# numerical schemes accumulate ∇·B errors at shocks and other features.
# MixedGLM augments the system with a 9th conserved scalar ψ:
#
#     ∂B/∂t + ∇·(uB − Bu) + ∇ψ      = 0                 ← B picks up a ψ-gradient
#     ∂ψ/∂t + c_h² · ∇·B           = −(c_h² / c_p²) · ψ ← ψ propagates ∇·B errors
#
# `c_h` is the hyperbolic cleaning speed (typically the maximum signal
# speed on the mesh); `c_p²` controls parabolic damping (smaller ⇒
# stronger damping). In conservative flux form:
#
#     f_x[B_x] = ψ
#     f_y[B_x] = u_y · B_x − B_y · u_x   (unchanged from IMHD)
#     f_x[B_y] = B_y · u_x − B_x · u_y   (unchanged)
#     f_y[B_y] = ψ
#     f_x[B_z] = u_x · B_z − B_x · u_z   (unchanged)
#     f_y[B_z] = u_y · B_z − B_y · u_z   (unchanged)
#     f_x[ψ]   = c_h² · B_x
#     f_y[ψ]   = c_h² · B_y
#
# Other components (ρ, momentum, E) use the standard IMHD fluxes.

from std.collections import InlineArray

from warpxm.apps.imhd_2d import (
    IMHD2D, MHD_NC,
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
    _fast_speed,
)
from warpxm.core.math_compat import fabs, fmax, sqrt
from warpxm.core.types import Real


comptime MHD_GLM_NC: Int = 9
comptime IDX_PSI: Int = 8


@fieldwise_init
struct ImhdGLM2D(ImplicitlyCopyable, Movable):
    """Ideal MHD + Dedner MixedGLM divergence cleaning, 2D."""

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
    # Internal flux helpers. We reuse the 8-component IMHD flux for the
    # MHD-physics components and add the ψ-coupling on top.
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_x_static(
        q: InlineArray[Real, MHD_GLM_NC],
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> InlineArray[Real, MHD_GLM_NC]:
        # Slice off the first 8 components to reuse the IMHD flux function.
        var q_mhd: InlineArray[Real, MHD_NC] = InlineArray[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            q_mhd[c] = q[c]
        var f_mhd = IMHD2D.internal_flux_x_static(q_mhd, gamma, rho_min, press_min)
        var f: InlineArray[Real, MHD_GLM_NC] = InlineArray[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            f[c] = f_mhd[c]
        # GLM coupling: ∂B_x/∂t += -∂ψ/∂x ⇒ f_x[B_x] = ψ.
        f[IDX_BX] = q[IDX_PSI]
        # ∂ψ/∂t += -c_h²·∇·B ⇒ f_x[ψ] = c_h² · B_x.
        f[IDX_PSI] = c_h * c_h * q[IDX_BX]
        return f

    @staticmethod
    def internal_flux_y_static(
        q: InlineArray[Real, MHD_GLM_NC],
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> InlineArray[Real, MHD_GLM_NC]:
        var q_mhd: InlineArray[Real, MHD_NC] = InlineArray[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            q_mhd[c] = q[c]
        var f_mhd = IMHD2D.internal_flux_y_static(q_mhd, gamma, rho_min, press_min)
        var f: InlineArray[Real, MHD_GLM_NC] = InlineArray[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            f[c] = f_mhd[c]
        f[IDX_BY] = q[IDX_PSI]
        f[IDX_PSI] = c_h * c_h * q[IDX_BY]
        return f

    # ------------------------------------------------------------------
    # Rusanov numerical flux. We take the max-signal speed to be the
    # larger of the IMHD fast magnetosonic speed and `c_h`.
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, MHD_GLM_NC],
        q_r: InlineArray[Real, MHD_GLM_NC],
        nx: Real, ny: Real,
        gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    ) -> InlineArray[Real, MHD_GLM_NC]:
        var rho_l = _rho_floored_glm(q_l, rho_min)
        var rho_r = _rho_floored_glm(q_r, rho_min)
        var p_l = _gas_pressure_floored_glm(q_l, gamma, rho_min, press_min)
        var p_r = _gas_pressure_floored_glm(q_r, gamma, rho_min, press_min)
        var bsq_l = q_l[IDX_BX] * q_l[IDX_BX] + q_l[IDX_BY] * q_l[IDX_BY] + q_l[IDX_BZ] * q_l[IDX_BZ]
        var bsq_r = q_r[IDX_BX] * q_r[IDX_BX] + q_r[IDX_BY] * q_r[IDX_BY] + q_r[IDX_BZ] * q_r[IDX_BZ]

        var u_n_l = (q_l[IDX_MX] * nx + q_l[IDX_MY] * ny) / rho_l
        var u_n_r = (q_r[IDX_MX] * nx + q_r[IDX_MY] * ny) / rho_r
        var B_n_l = q_l[IDX_BX] * nx + q_l[IDX_BY] * ny
        var B_n_r = q_r[IDX_BX] * nx + q_r[IDX_BY] * ny

        var c_f_l = _fast_speed(rho_l, p_l, B_n_l, bsq_l, gamma)
        var c_f_r = _fast_speed(rho_r, p_r, B_n_r, bsq_r, gamma)
        var c_max_mhd = fmax(fabs(u_n_l) + c_f_l, fabs(u_n_r) + c_f_r)
        var c_max = fmax(c_max_mhd, c_h)

        var fx_l = Self.internal_flux_x_static(q_l, gamma, rho_min, press_min, c_h)
        var fy_l = Self.internal_flux_y_static(q_l, gamma, rho_min, press_min, c_h)
        var fx_r = Self.internal_flux_x_static(q_r, gamma, rho_min, press_min, c_h)
        var fy_r = Self.internal_flux_y_static(q_r, gamma, rho_min, press_min, c_h)

        var f: InlineArray[Real, MHD_GLM_NC] = InlineArray[Real, MHD_GLM_NC](fill=Real(0.0))
        comptime for c in range(MHD_GLM_NC):
            var f_n_l = nx * fx_l[c] + ny * fy_l[c]
            var f_n_r = nx * fx_r[c] + ny * fy_r[c]
            f[c] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f


# ---------------------------------------------------------------------------
# GLM-flavoured versions of the IMHD helper functions (operate on 9-component
# arrays rather than 8). They just ignore q[IDX_PSI] for the MHD-physics
# calculations.
# ---------------------------------------------------------------------------

@always_inline
def _rho_floored_glm(q: InlineArray[Real, MHD_GLM_NC], rho_min: Real) -> Real:
    return fmax(q[IDX_RHO], rho_min)


@always_inline
def _gas_pressure_floored_glm(
    q: InlineArray[Real, MHD_GLM_NC], gamma: Real, rho_min: Real, press_min: Real,
) -> Real:
    var rho = _rho_floored_glm(q, rho_min)
    var msq = (q[IDX_MX] * q[IDX_MX]
               + q[IDX_MY] * q[IDX_MY]
               + q[IDX_MZ] * q[IDX_MZ])
    var ke = Real(0.5) * msq / rho
    var bsq = q[IDX_BX] * q[IDX_BX] + q[IDX_BY] * q[IDX_BY] + q[IDX_BZ] * q[IDX_BZ]
    var mag = Real(0.5) * bsq
    var p = (gamma - Real(1.0)) * (q[IDX_E] - ke - mag)
    return fmax(p, press_min)
