# Ideal MHD in 2D.
#
# 8 conserved variables: q = (ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z).
# γ-law EOS, μ₀ = 1 in normalised units.
#
# Total energy density:
#
#     E = P_gas / (γ − 1) + ½ · ρ · |u|² + ½ · |B|²
#
# Gas pressure recovered from conserved state:
#
#     P_gas = (γ − 1) · ( E − ½ · |m|² / ρ − ½ · |B|² )
#
# Flux tensor (x-component shown; y is the symmetric counterpart):
#
#     f_x[ρ]   = ρu_x
#     f_x[m_x] = ρu_x² + P_gas + |B|²/2 − B_x²
#     f_x[m_y] = ρu_x · u_y − B_x · B_y
#     f_x[m_z] = ρu_x · u_z − B_x · B_z
#     f_x[E]   = (E + P_gas + |B|²/2) · u_x − B_x · (u·B)
#     f_x[B_x] = 0                                          ← B_x is frozen on x-faces
#     f_x[B_y] = B_y · u_x − B_x · u_y
#     f_x[B_z] = B_z · u_x − B_x · u_z
#
# Numerical flux: Rusanov (local Lax-Friedrichs). The maximum signal speed
# is the fast magnetosonic speed `c_f = sqrt(½ (c_s² + c_a²) + ½ sqrt(...))`
# evaluated in the face-normal direction. See `_fast_speed` below for the
# closed form.

from std.collections import InlineArray

from warpxm.core.math_compat import fabs, fmax, sqrt
from warpxm.core.types import Real


comptime MHD_NC: Int = 8

# Conserved-variable indices.
comptime IDX_RHO: Int = 0
comptime IDX_MX: Int = 1
comptime IDX_MY: Int = 2
comptime IDX_MZ: Int = 3
comptime IDX_E: Int = 4
comptime IDX_BX: Int = 5
comptime IDX_BY: Int = 6
comptime IDX_BZ: Int = 7


@always_inline
def _rho_floored(q: InlineArray[Real, MHD_NC], rho_min: Real) -> Real:
    return fmax(q[IDX_RHO], rho_min)


@always_inline
def _bsq(q: InlineArray[Real, MHD_NC]) -> Real:
    return q[IDX_BX] * q[IDX_BX] + q[IDX_BY] * q[IDX_BY] + q[IDX_BZ] * q[IDX_BZ]


@always_inline
def _gas_pressure_floored(
    q: InlineArray[Real, MHD_NC], gamma: Real, rho_min: Real, press_min: Real,
) -> Real:
    var rho = _rho_floored(q, rho_min)
    var msq = (q[IDX_MX] * q[IDX_MX]
               + q[IDX_MY] * q[IDX_MY]
               + q[IDX_MZ] * q[IDX_MZ])
    var ke = Real(0.5) * msq / rho
    var mag = Real(0.5) * _bsq(q)
    var p = (gamma - Real(1.0)) * (q[IDX_E] - ke - mag)
    return fmax(p, press_min)


@always_inline
def _fast_speed(
    rho: Real, p_gas: Real, Bn: Real, Btot_sq: Real, gamma: Real,
) -> Real:
    """Fast magnetosonic speed in the direction of normal component `B_n`.

    c_f² = ½ (c_s² + c_a²) + ½ √( (c_s² + c_a²)² − 4 c_s² c_a_n² )

    Where c_s² = γ P / ρ (gas sound), c_a² = |B|²/ρ (Alfvén), and
    c_a_n² = B_n²/ρ is the Alfvén speed along the normal.
    """
    var inv_rho = Real(1.0) / rho
    var c_s_sq = gamma * p_gas * inv_rho
    var c_a_sq = Btot_sq * inv_rho
    var c_a_n_sq = Bn * Bn * inv_rho
    var sum_sq = c_s_sq + c_a_sq
    var disc = sum_sq * sum_sq - Real(4.0) * c_s_sq * c_a_n_sq
    var disc_safe = fmax(disc, Real(0.0))
    return sqrt(Real(0.5) * (sum_sq + sqrt(disc_safe)))


@fieldwise_init
struct IMHD2D(ImplicitlyCopyable, Movable):
    """Ideal MHD in 2D with γ-law EOS and Rusanov numerical flux."""

    var gas_gamma: Real
    var min_density: Real
    var min_pressure: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return MHD_NC

    # ------------------------------------------------------------------
    # Internal flux, x and y components. Both share most of the algebra;
    # we keep two thin functions instead of one to match the pattern of
    # `apps/euler_2d.mojo` and `apps/maxwell_2d.mojo`.
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_x_static(
        q: InlineArray[Real, MHD_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, MHD_NC]:
        var rho = _rho_floored(q, rho_min)
        var p = _gas_pressure_floored(q, gamma, rho_min, press_min)
        var ux = q[IDX_MX] / rho
        var uy = q[IDX_MY] / rho
        var uz = q[IDX_MZ] / rho
        var Bx = q[IDX_BX]
        var By = q[IDX_BY]
        var Bz = q[IDX_BZ]
        var bsq = _bsq(q)
        var udotB = ux * Bx + uy * By + uz * Bz
        var p_tot = p + Real(0.5) * bsq

        var f: InlineArray[Real, MHD_NC] = InlineArray[Real, MHD_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MX]
        f[IDX_MX] = q[IDX_MX] * ux + p_tot - Bx * Bx
        f[IDX_MY] = q[IDX_MX] * uy - Bx * By
        f[IDX_MZ] = q[IDX_MX] * uz - Bx * Bz
        f[IDX_E] = (q[IDX_E] + p_tot) * ux - Bx * udotB
        f[IDX_BX] = Real(0.0)
        f[IDX_BY] = By * ux - Bx * uy
        f[IDX_BZ] = Bz * ux - Bx * uz
        return f

    @staticmethod
    def internal_flux_y_static(
        q: InlineArray[Real, MHD_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, MHD_NC]:
        var rho = _rho_floored(q, rho_min)
        var p = _gas_pressure_floored(q, gamma, rho_min, press_min)
        var ux = q[IDX_MX] / rho
        var uy = q[IDX_MY] / rho
        var uz = q[IDX_MZ] / rho
        var Bx = q[IDX_BX]
        var By = q[IDX_BY]
        var Bz = q[IDX_BZ]
        var bsq = _bsq(q)
        var udotB = ux * Bx + uy * By + uz * Bz
        var p_tot = p + Real(0.5) * bsq

        var f: InlineArray[Real, MHD_NC] = InlineArray[Real, MHD_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MY]
        f[IDX_MX] = q[IDX_MY] * ux - By * Bx
        f[IDX_MY] = q[IDX_MY] * uy + p_tot - By * By
        f[IDX_MZ] = q[IDX_MY] * uz - By * Bz
        f[IDX_E] = (q[IDX_E] + p_tot) * uy - By * udotB
        f[IDX_BX] = Bx * uy - By * ux
        f[IDX_BY] = Real(0.0)
        f[IDX_BZ] = Bz * uy - By * uz
        return f

    # ------------------------------------------------------------------
    # Rusanov numerical flux on an arbitrary unit normal (nx, ny).
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, MHD_NC],
        q_r: InlineArray[Real, MHD_NC],
        nx: Real,
        ny: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, MHD_NC]:
        var rho_l = _rho_floored(q_l, rho_min)
        var rho_r = _rho_floored(q_r, rho_min)
        var p_l = _gas_pressure_floored(q_l, gamma, rho_min, press_min)
        var p_r = _gas_pressure_floored(q_r, gamma, rho_min, press_min)
        var bsq_l = _bsq(q_l)
        var bsq_r = _bsq(q_r)

        var u_n_l = (q_l[IDX_MX] * nx + q_l[IDX_MY] * ny) / rho_l
        var u_n_r = (q_r[IDX_MX] * nx + q_r[IDX_MY] * ny) / rho_r
        var B_n_l = q_l[IDX_BX] * nx + q_l[IDX_BY] * ny
        var B_n_r = q_r[IDX_BX] * nx + q_r[IDX_BY] * ny

        var c_f_l = _fast_speed(rho_l, p_l, B_n_l, bsq_l, gamma)
        var c_f_r = _fast_speed(rho_r, p_r, B_n_r, bsq_r, gamma)
        var c_max = fmax(fabs(u_n_l) + c_f_l, fabs(u_n_r) + c_f_r)

        var fx_l = Self.internal_flux_x_static(q_l, gamma, rho_min, press_min)
        var fy_l = Self.internal_flux_y_static(q_l, gamma, rho_min, press_min)
        var fx_r = Self.internal_flux_x_static(q_r, gamma, rho_min, press_min)
        var fy_r = Self.internal_flux_y_static(q_r, gamma, rho_min, press_min)

        var f: InlineArray[Real, MHD_NC] = InlineArray[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            var f_n_l = nx * fx_l[c] + ny * fy_l[c]
            var f_n_r = nx * fx_r[c] + ny * fy_r[c]
            f[c] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f

    # ------------------------------------------------------------------
    # HLLD numerical flux (Miyoshi & Kusano, 2005).
    #
    # Five-state approximate Riemann solver resolving fast magnetosonic
    # (S_L, S_R), Alfvén (S_L*, S_R*), and contact / entropy (S_M) waves.
    # Sharper shocks and rotational discontinuities than Rusanov (LF).
    #
    # Algorithm:
    #   * Rotate L/R states to face-normal frame: u_n, u_t1, u_t2; B_n,
    #     B_t1, B_t2.  Here t1 is the in-plane tangent (-ny, nx), t2 = ẑ.
    #   * Compute fast speeds c_f and S_L = min(u_n)-max(c_f),
    #     S_R = max(u_n)+max(c_f).
    #   * Middle wave speed S_M from Rankine-Hugoniot on the contact.
    #   * Star (HLL) and double-star (Alfvén) states between (S_L, S_M)
    #     and (S_M, S_R).  Fluxes via U-state jumps:
    #         F* = F + S·(U* - U), F** = F* + S*·(U** - U*).
    #   * Select region containing s=0 and return that flux.
    #
    # Degenerate case B_n ≈ 0 collapses HLLD to HLLC for MHD (Alfvén
    # speeds coincide with S_M); we use a small floor to keep the
    # division stable.
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_hlld_static(
        q_l: InlineArray[Real, MHD_NC],
        q_r: InlineArray[Real, MHD_NC],
        nx: Real,
        ny: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, MHD_NC]:
        # In-plane tangent t = (-ny, nx).
        var tx = -ny
        var ty = nx

        # --- L state: rotate to (n, t, z) frame -------------------------
        var rho_l = _rho_floored(q_l, rho_min)
        var p_gas_l = _gas_pressure_floored(q_l, gamma, rho_min, press_min)
        var bsq_l = _bsq(q_l)
        var ux_l = q_l[IDX_MX] / rho_l
        var uy_l = q_l[IDX_MY] / rho_l
        var uz_l = q_l[IDX_MZ] / rho_l
        var u_n_l = ux_l * nx + uy_l * ny
        var u_t_l = ux_l * tx + uy_l * ty
        var B_n_l = q_l[IDX_BX] * nx + q_l[IDX_BY] * ny
        var B_t_l = q_l[IDX_BX] * tx + q_l[IDX_BY] * ty
        var B_z_l = q_l[IDX_BZ]
        var p_T_l = p_gas_l + Real(0.5) * bsq_l
        var c_f_l = _fast_speed(rho_l, p_gas_l, B_n_l, bsq_l, gamma)

        # --- R state ----------------------------------------------------
        var rho_r = _rho_floored(q_r, rho_min)
        var p_gas_r = _gas_pressure_floored(q_r, gamma, rho_min, press_min)
        var bsq_r = _bsq(q_r)
        var ux_r = q_r[IDX_MX] / rho_r
        var uy_r = q_r[IDX_MY] / rho_r
        var uz_r = q_r[IDX_MZ] / rho_r
        var u_n_r = ux_r * nx + uy_r * ny
        var u_t_r = ux_r * tx + uy_r * ty
        var B_n_r = q_r[IDX_BX] * nx + q_r[IDX_BY] * ny
        var B_t_r = q_r[IDX_BX] * tx + q_r[IDX_BY] * ty
        var B_z_r = q_r[IDX_BZ]
        var p_T_r = p_gas_r + Real(0.5) * bsq_r
        var c_f_r = _fast_speed(rho_r, p_gas_r, B_n_r, bsq_r, gamma)

        # --- Outer wave speeds (Davis 1988 estimates) ------------------
        var c_max = fmax(c_f_l, c_f_r)
        var S_L = fmin_real(u_n_l, u_n_r) - c_max
        var S_R = fmax_real(u_n_l, u_n_r) + c_max

        # B_n must be continuous across the interface; HLLD assumes this.
        # Use the average for robustness.
        var B_n = Real(0.5) * (B_n_l + B_n_r)

        # --- Build rotated conserved-state vectors ---------------------
        # U = (ρ, ρu_n, ρu_t, ρu_z, E, B_n, B_t, B_z) in rotated frame.
        var U_L = InlineArray[Real, MHD_NC](fill=Real(0.0))
        U_L[IDX_RHO] = rho_l
        U_L[IDX_MX]  = rho_l * u_n_l
        U_L[IDX_MY]  = rho_l * u_t_l
        U_L[IDX_MZ]  = rho_l * uz_l
        U_L[IDX_E]   = q_l[IDX_E]
        U_L[IDX_BX]  = B_n
        U_L[IDX_BY]  = B_t_l
        U_L[IDX_BZ]  = B_z_l
        var U_R = InlineArray[Real, MHD_NC](fill=Real(0.0))
        U_R[IDX_RHO] = rho_r
        U_R[IDX_MX]  = rho_r * u_n_r
        U_R[IDX_MY]  = rho_r * u_t_r
        U_R[IDX_MZ]  = rho_r * uz_r
        U_R[IDX_E]   = q_r[IDX_E]
        U_R[IDX_BX]  = B_n
        U_R[IDX_BY]  = B_t_r
        U_R[IDX_BZ]  = B_z_r

        # Normal-direction fluxes in rotated frame.
        # F[ρ]   = ρu_n
        # F[m_n] = ρu_n² + p_T - B_n²
        # F[m_t] = ρu_n u_t - B_n B_t
        # F[m_z] = ρu_n u_z - B_n B_z
        # F[E]   = (E + p_T) u_n - B_n (u·B)
        # F[B_n] = 0
        # F[B_t] = B_t u_n - B_n u_t
        # F[B_z] = B_z u_n - B_n u_z
        var udotB_l = u_n_l * B_n + u_t_l * B_t_l + uz_l * B_z_l
        var udotB_r = u_n_r * B_n + u_t_r * B_t_r + uz_r * B_z_r
        var F_L = InlineArray[Real, MHD_NC](fill=Real(0.0))
        F_L[IDX_RHO] = rho_l * u_n_l
        F_L[IDX_MX]  = rho_l * u_n_l * u_n_l + p_T_l - B_n * B_n
        F_L[IDX_MY]  = rho_l * u_n_l * u_t_l - B_n * B_t_l
        F_L[IDX_MZ]  = rho_l * u_n_l * uz_l - B_n * B_z_l
        F_L[IDX_E]   = (q_l[IDX_E] + p_T_l) * u_n_l - B_n * udotB_l
        F_L[IDX_BX]  = Real(0.0)
        F_L[IDX_BY]  = B_t_l * u_n_l - B_n * u_t_l
        F_L[IDX_BZ]  = B_z_l * u_n_l - B_n * uz_l
        var F_R = InlineArray[Real, MHD_NC](fill=Real(0.0))
        F_R[IDX_RHO] = rho_r * u_n_r
        F_R[IDX_MX]  = rho_r * u_n_r * u_n_r + p_T_r - B_n * B_n
        F_R[IDX_MY]  = rho_r * u_n_r * u_t_r - B_n * B_t_r
        F_R[IDX_MZ]  = rho_r * u_n_r * uz_r - B_n * B_z_r
        F_R[IDX_E]   = (q_r[IDX_E] + p_T_r) * u_n_r - B_n * udotB_r
        F_R[IDX_BX]  = Real(0.0)
        F_R[IDX_BY]  = B_t_r * u_n_r - B_n * u_t_r
        F_R[IDX_BZ]  = B_z_r * u_n_r - B_n * uz_r

        # Early outs.
        if S_L >= Real(0.0):
            return Self._rotate_flux_back(F_L, nx, ny)
        if S_R <= Real(0.0):
            return Self._rotate_flux_back(F_R, nx, ny)

        # --- Middle wave speed S_M --------------------------------------
        var sL_uL = S_L - u_n_l
        var sR_uR = S_R - u_n_r
        var num_sm = sR_uR * rho_r * u_n_r - sL_uL * rho_l * u_n_l - p_T_r + p_T_l
        var den_sm = sR_uR * rho_r - sL_uL * rho_l
        var S_M = num_sm / den_sm
        var p_T_star = p_T_l + rho_l * sL_uL * (S_M - u_n_l)

        # --- Star states (between S_L and S_M, S_M and S_R) -------------
        var rho_L_s = rho_l * sL_uL / (S_L - S_M)
        var rho_R_s = rho_r * sR_uR / (S_R - S_M)
        var abs_Bn = fabs(B_n)
        var S_L_s = S_M - abs_Bn / sqrt(rho_L_s)
        var S_R_s = S_M + abs_Bn / sqrt(rho_R_s)

        # Star transverse states (B_n ≠ 0 path).  den_K = ρ_K·sLR_uLR·(S_K - S_M) - B_n²
        # ≈ ρ_K·sLR_uLR·(S_K - S_M) for B_n strong; → 0 for B_n → 0.
        # Use a safe floor; degenerate case yields B_t* = B_t, u_t* = u_t.
        var EPS = Real(1.0e-30)

        var den_L = rho_l * sL_uL * (S_L - S_M) - B_n * B_n
        var u_t_L_s: Real
        var u_z_L_s: Real
        var B_t_L_s: Real
        var B_z_L_s: Real
        if fabs(den_L) < EPS:
            u_t_L_s = u_t_l
            u_z_L_s = uz_l
            B_t_L_s = B_t_l
            B_z_L_s = B_z_l
        else:
            var inv_den_L = Real(1.0) / den_L
            u_t_L_s = u_t_l - B_n * B_t_l * (S_M - u_n_l) * inv_den_L
            u_z_L_s = uz_l - B_n * B_z_l * (S_M - u_n_l) * inv_den_L
            var num_BL = rho_l * sL_uL * sL_uL - B_n * B_n
            B_t_L_s = B_t_l * num_BL * inv_den_L
            B_z_L_s = B_z_l * num_BL * inv_den_L

        var den_R = rho_r * sR_uR * (S_R - S_M) - B_n * B_n
        var u_t_R_s: Real
        var u_z_R_s: Real
        var B_t_R_s: Real
        var B_z_R_s: Real
        if fabs(den_R) < EPS:
            u_t_R_s = u_t_r
            u_z_R_s = uz_r
            B_t_R_s = B_t_r
            B_z_R_s = B_z_r
        else:
            var inv_den_R = Real(1.0) / den_R
            u_t_R_s = u_t_r - B_n * B_t_r * (S_M - u_n_r) * inv_den_R
            u_z_R_s = uz_r - B_n * B_z_r * (S_M - u_n_r) * inv_den_R
            var num_BR = rho_r * sR_uR * sR_uR - B_n * B_n
            B_t_R_s = B_t_r * num_BR * inv_den_R
            B_z_R_s = B_z_r * num_BR * inv_den_R

        var udotB_L_s = S_M * B_n + u_t_L_s * B_t_L_s + u_z_L_s * B_z_L_s
        var udotB_R_s = S_M * B_n + u_t_R_s * B_t_R_s + u_z_R_s * B_z_R_s
        var E_L_s = (sL_uL * q_l[IDX_E] - p_T_l * u_n_l + p_T_star * S_M
                     + B_n * (udotB_l - udotB_L_s)) / (S_L - S_M)
        var E_R_s = (sR_uR * q_r[IDX_E] - p_T_r * u_n_r + p_T_star * S_M
                     + B_n * (udotB_r - udotB_R_s)) / (S_R - S_M)

        # Build star U vectors.
        var U_L_s = InlineArray[Real, MHD_NC](fill=Real(0.0))
        U_L_s[IDX_RHO] = rho_L_s
        U_L_s[IDX_MX]  = rho_L_s * S_M
        U_L_s[IDX_MY]  = rho_L_s * u_t_L_s
        U_L_s[IDX_MZ]  = rho_L_s * u_z_L_s
        U_L_s[IDX_E]   = E_L_s
        U_L_s[IDX_BX]  = B_n
        U_L_s[IDX_BY]  = B_t_L_s
        U_L_s[IDX_BZ]  = B_z_L_s
        var U_R_s = InlineArray[Real, MHD_NC](fill=Real(0.0))
        U_R_s[IDX_RHO] = rho_R_s
        U_R_s[IDX_MX]  = rho_R_s * S_M
        U_R_s[IDX_MY]  = rho_R_s * u_t_R_s
        U_R_s[IDX_MZ]  = rho_R_s * u_z_R_s
        U_R_s[IDX_E]   = E_R_s
        U_R_s[IDX_BX]  = B_n
        U_R_s[IDX_BY]  = B_t_R_s
        U_R_s[IDX_BZ]  = B_z_R_s

        # F* = F + S·(U* - U)
        var F_L_s = InlineArray[Real, MHD_NC](fill=Real(0.0))
        var F_R_s = InlineArray[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            F_L_s[c] = F_L[c] + S_L * (U_L_s[c] - U_L[c])
            F_R_s[c] = F_R[c] + S_R * (U_R_s[c] - U_R[c])

        if S_L_s >= Real(0.0):
            return Self._rotate_flux_back(F_L_s, nx, ny)
        if S_R_s <= Real(0.0):
            return Self._rotate_flux_back(F_R_s, nx, ny)

        # --- Double-star (between S_L* and S_M, S_M and S_R*) -----------
        var sqrt_rL = sqrt(rho_L_s)
        var sqrt_rR = sqrt(rho_R_s)
        var denom_ss = sqrt_rL + sqrt_rR
        var sign_Bn: Real = 1.0 if B_n >= Real(0.0) else -1.0
        var u_t_ss = (sqrt_rL * u_t_L_s + sqrt_rR * u_t_R_s
                      + (B_t_R_s - B_t_L_s) * sign_Bn) / denom_ss
        var u_z_ss = (sqrt_rL * u_z_L_s + sqrt_rR * u_z_R_s
                      + (B_z_R_s - B_z_L_s) * sign_Bn) / denom_ss
        var B_t_ss = (sqrt_rL * B_t_R_s + sqrt_rR * B_t_L_s
                      + sqrt_rL * sqrt_rR * (u_t_R_s - u_t_L_s) * sign_Bn) / denom_ss
        var B_z_ss = (sqrt_rL * B_z_R_s + sqrt_rR * B_z_L_s
                      + sqrt_rL * sqrt_rR * (u_z_R_s - u_z_L_s) * sign_Bn) / denom_ss
        var udotB_ss = S_M * B_n + u_t_ss * B_t_ss + u_z_ss * B_z_ss
        var E_L_ss = E_L_s - sqrt_rL * sign_Bn * (udotB_L_s - udotB_ss)
        var E_R_ss = E_R_s + sqrt_rR * sign_Bn * (udotB_R_s - udotB_ss)

        # Build double-star U vectors.
        var U_L_ss = InlineArray[Real, MHD_NC](fill=Real(0.0))
        U_L_ss[IDX_RHO] = rho_L_s
        U_L_ss[IDX_MX]  = rho_L_s * S_M
        U_L_ss[IDX_MY]  = rho_L_s * u_t_ss
        U_L_ss[IDX_MZ]  = rho_L_s * u_z_ss
        U_L_ss[IDX_E]   = E_L_ss
        U_L_ss[IDX_BX]  = B_n
        U_L_ss[IDX_BY]  = B_t_ss
        U_L_ss[IDX_BZ]  = B_z_ss
        var U_R_ss = InlineArray[Real, MHD_NC](fill=Real(0.0))
        U_R_ss[IDX_RHO] = rho_R_s
        U_R_ss[IDX_MX]  = rho_R_s * S_M
        U_R_ss[IDX_MY]  = rho_R_s * u_t_ss
        U_R_ss[IDX_MZ]  = rho_R_s * u_z_ss
        U_R_ss[IDX_E]   = E_R_ss
        U_R_ss[IDX_BX]  = B_n
        U_R_ss[IDX_BY]  = B_t_ss
        U_R_ss[IDX_BZ]  = B_z_ss

        var F_chosen = InlineArray[Real, MHD_NC](fill=Real(0.0))
        if S_M >= Real(0.0):
            comptime for c in range(MHD_NC):
                F_chosen[c] = F_L_s[c] + S_L_s * (U_L_ss[c] - U_L_s[c])
        else:
            comptime for c in range(MHD_NC):
                F_chosen[c] = F_R_s[c] + S_R_s * (U_R_ss[c] - U_R_s[c])

        return Self._rotate_flux_back(F_chosen, nx, ny)

    @staticmethod
    def _rotate_flux_back(
        F_rot: InlineArray[Real, MHD_NC], nx: Real, ny: Real,
    ) -> InlineArray[Real, MHD_NC]:
        """Rotate a flux computed in face-normal frame back to (x, y, z)."""
        var tx = -ny
        var ty = nx
        var out = InlineArray[Real, MHD_NC](fill=Real(0.0))
        out[IDX_RHO] = F_rot[IDX_RHO]
        # Momentum: f_mx = nx·f_mn + tx·f_mt
        out[IDX_MX] = nx * F_rot[IDX_MX] + tx * F_rot[IDX_MY]
        out[IDX_MY] = ny * F_rot[IDX_MX] + ty * F_rot[IDX_MY]
        out[IDX_MZ] = F_rot[IDX_MZ]
        out[IDX_E] = F_rot[IDX_E]
        # B: f_Bx = nx·f_Bn + tx·f_Bt
        out[IDX_BX] = nx * F_rot[IDX_BX] + tx * F_rot[IDX_BY]
        out[IDX_BY] = ny * F_rot[IDX_BX] + ty * F_rot[IDX_BY]
        out[IDX_BZ] = F_rot[IDX_BZ]
        return out


@always_inline
def fmin_real(a: Real, b: Real) -> Real:
    return a if a < b else b


@always_inline
def fmax_real(a: Real, b: Real) -> Real:
    return a if a > b else b
