# Ideal MHD — single dimension-agnostic app.
#
# 8 conserved variables: q = (ρ, ρu_x, ρu_y, ρu_z, E, B_x, B_y, B_z).
# γ-law EOS, μ₀ = 1 in normalised units.
#
# The 2D in-plane-rotation HLLD has been replaced by the 3D axis-aligned
# permutation-based core (`_hlld_core_3d`); the DG launcher always passes
# axis-aligned face normals so a single dispatch path handles all
# (nx, ny, nz) ∈ {(±1,0,0), (0,±1,0), (0,0,±1)}.
#
# Numerical fluxes:
#   * Rusanov / local Lax-Friedrichs (default).
#   * Miyoshi-Kusano HLLD (5-state approximate Riemann solver).

from std.collections import Array

from warpxm.core.math_compat import fabs, fmax, sqrt
from warpxm.core.types import Real


comptime MHD_NC: Int = 8

comptime IDX_RHO: Int = 0
comptime IDX_MX: Int = 1
comptime IDX_MY: Int = 2
comptime IDX_MZ: Int = 3
comptime IDX_E: Int = 4
comptime IDX_BX: Int = 5
comptime IDX_BY: Int = 6
comptime IDX_BZ: Int = 7


@always_inline
def _rho_floored(q: Array[Real, MHD_NC], rho_min: Real) -> Real:
    return fmax(q[IDX_RHO], rho_min)


@always_inline
def _bsq(q: Array[Real, MHD_NC]) -> Real:
    return q[IDX_BX] * q[IDX_BX] + q[IDX_BY] * q[IDX_BY] + q[IDX_BZ] * q[IDX_BZ]


@always_inline
def _gas_pressure_floored(
    q: Array[Real, MHD_NC], gamma: Real, rho_min: Real, press_min: Real,
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
    """Fast magnetosonic speed in the direction of normal component `B_n`."""
    var inv_rho = Real(1.0) / rho
    var c_s_sq = gamma * p_gas * inv_rho
    var c_a_sq = Btot_sq * inv_rho
    var c_a_n_sq = Bn * Bn * inv_rho
    var sum_sq = c_s_sq + c_a_sq
    var disc = sum_sq * sum_sq - Real(4.0) * c_s_sq * c_a_n_sq
    var disc_safe = fmax(disc, Real(0.0))
    return sqrt(Real(0.5) * (sum_sq + sqrt(disc_safe)))


@fieldwise_init
struct IMHD(ImplicitlyCopyable, Movable):
    """Ideal MHD with γ-law EOS, Rusanov + HLLD numerical fluxes."""

    var gas_gamma: Real
    var min_density: Real
    var min_pressure: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return MHD_NC

    @staticmethod
    def internal_flux_x_static(
        q: Array[Real, MHD_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, MHD_NC]:
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

        var f: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MX]
        f[IDX_MX] = q[IDX_MX] * ux + p_tot - Bx * Bx
        f[IDX_MY] = q[IDX_MX] * uy - Bx * By
        f[IDX_MZ] = q[IDX_MX] * uz - Bx * Bz
        f[IDX_E] = (q[IDX_E] + p_tot) * ux - Bx * udotB
        f[IDX_BX] = Real(0.0)
        f[IDX_BY] = By * ux - Bx * uy
        f[IDX_BZ] = Bz * ux - Bx * uz
        return f^

    @staticmethod
    def internal_flux_y_static(
        q: Array[Real, MHD_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, MHD_NC]:
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

        var f: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MY]
        f[IDX_MX] = q[IDX_MY] * ux - By * Bx
        f[IDX_MY] = q[IDX_MY] * uy + p_tot - By * By
        f[IDX_MZ] = q[IDX_MY] * uz - By * Bz
        f[IDX_E] = (q[IDX_E] + p_tot) * uy - By * udotB
        f[IDX_BX] = Bx * uy - By * ux
        f[IDX_BY] = Real(0.0)
        f[IDX_BZ] = Bz * uy - By * uz
        return f^

    @staticmethod
    def internal_flux_z_static(
        q: Array[Real, MHD_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, MHD_NC]:
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

        var f: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MZ]
        f[IDX_MX] = q[IDX_MZ] * ux - Bz * Bx
        f[IDX_MY] = q[IDX_MZ] * uy - Bz * By
        f[IDX_MZ] = q[IDX_MZ] * uz + p_tot - Bz * Bz
        f[IDX_E] = (q[IDX_E] + p_tot) * uz - Bz * udotB
        f[IDX_BX] = Bx * uz - Bz * ux
        f[IDX_BY] = By * uz - Bz * uy
        f[IDX_BZ] = Real(0.0)
        return f^

    # ------------------------------------------------------------------
    # Rusanov numerical flux on a 3-component unit normal.
    # 2D callers pass (nx, ny, 0).
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: Array[Real, MHD_NC],
        q_r: Array[Real, MHD_NC],
        nx: Real, ny: Real, nz: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, MHD_NC]:
        var rho_l = _rho_floored(q_l, rho_min)
        var rho_r = _rho_floored(q_r, rho_min)
        var p_l = _gas_pressure_floored(q_l, gamma, rho_min, press_min)
        var p_r = _gas_pressure_floored(q_r, gamma, rho_min, press_min)
        var bsq_l = _bsq(q_l)
        var bsq_r = _bsq(q_r)

        var u_n_l = (q_l[IDX_MX] * nx + q_l[IDX_MY] * ny + q_l[IDX_MZ] * nz) / rho_l
        var u_n_r = (q_r[IDX_MX] * nx + q_r[IDX_MY] * ny + q_r[IDX_MZ] * nz) / rho_r
        var B_n_l = q_l[IDX_BX] * nx + q_l[IDX_BY] * ny + q_l[IDX_BZ] * nz
        var B_n_r = q_r[IDX_BX] * nx + q_r[IDX_BY] * ny + q_r[IDX_BZ] * nz

        var c_f_l = _fast_speed(rho_l, p_l, B_n_l, bsq_l, gamma)
        var c_f_r = _fast_speed(rho_r, p_r, B_n_r, bsq_r, gamma)
        var c_max = fmax(fabs(u_n_l) + c_f_l, fabs(u_n_r) + c_f_r)

        var fx_l = Self.internal_flux_x_static(q_l, gamma, rho_min, press_min)
        var fy_l = Self.internal_flux_y_static(q_l, gamma, rho_min, press_min)
        var fz_l = Self.internal_flux_z_static(q_l, gamma, rho_min, press_min)
        var fx_r = Self.internal_flux_x_static(q_r, gamma, rho_min, press_min)
        var fy_r = Self.internal_flux_y_static(q_r, gamma, rho_min, press_min)
        var fz_r = Self.internal_flux_z_static(q_r, gamma, rho_min, press_min)

        var f: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
        comptime for c in range(MHD_NC):
            var f_n_l = nx * fx_l[c] + ny * fy_l[c] + nz * fz_l[c]
            var f_n_r = nx * fx_r[c] + ny * fy_r[c] + nz * fz_r[c]
            f[c] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f^

    # ------------------------------------------------------------------
    # HLLD numerical flux (Miyoshi-Kusano 2005) on an axis-aligned face.
    # The DG launcher always passes one of (±1,0,0), (0,±1,0), (0,0,±1).
    # Each axis case dispatches to `_hlld_core_3d` with the right
    # (normal, tangent₁, tangent₂) right-handed permutation:
    #
    #   x-normal:  (n, t₁, t₂) = (x, y, z)
    #   y-normal:  (n, t₁, t₂) = (y, z, x)
    #   z-normal:  (n, t₁, t₂) = (z, x, y)
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_hlld_static(
        q_l: Array[Real, MHD_NC],
        q_r: Array[Real, MHD_NC],
        nx: Real, ny: Real, nz: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, MHD_NC]:
        if fabs(nx) > Real(0.5):
            return _hlld_core_3d(
                q_l, q_r,
                IDX_MX, IDX_MY, IDX_MZ, IDX_BX, IDX_BY, IDX_BZ,
                gamma, rho_min, press_min,
            )
        if fabs(ny) > Real(0.5):
            return _hlld_core_3d(
                q_l, q_r,
                IDX_MY, IDX_MZ, IDX_MX, IDX_BY, IDX_BZ, IDX_BX,
                gamma, rho_min, press_min,
            )
        # z-normal
        return _hlld_core_3d(
            q_l, q_r,
            IDX_MZ, IDX_MX, IDX_MY, IDX_BZ, IDX_BX, IDX_BY,
            gamma, rho_min, press_min,
        )


def _hlld_core_3d(
    q_l: Array[Real, MHD_NC],
    q_r: Array[Real, MHD_NC],
    M_N: Int, M_T1: Int, M_T2: Int,
    B_N: Int, B_T1: Int, B_T2: Int,
    gamma: Real, rho_min: Real, press_min: Real,
) -> Array[Real, MHD_NC]:
    """Miyoshi-Kusano HLLD in the (n, t₁, t₂) frame. Index parameters
    pick which conserved-variable component is the normal momentum, the
    two transverse momenta, the normal B, and the two transverse Bs.
    """
    var rho_l = _rho_floored(q_l, rho_min)
    var p_gas_l = _gas_pressure_floored(q_l, gamma, rho_min, press_min)
    var bsq_l = _bsq(q_l)
    var u_n_l = q_l[M_N] / rho_l
    var u_t1_l = q_l[M_T1] / rho_l
    var u_t2_l = q_l[M_T2] / rho_l
    var B_n_l = q_l[B_N]
    var B_t1_l = q_l[B_T1]
    var B_t2_l = q_l[B_T2]
    var p_T_l = p_gas_l + Real(0.5) * bsq_l
    var c_f_l = _fast_speed(rho_l, p_gas_l, B_n_l, bsq_l, gamma)

    var rho_r = _rho_floored(q_r, rho_min)
    var p_gas_r = _gas_pressure_floored(q_r, gamma, rho_min, press_min)
    var bsq_r = _bsq(q_r)
    var u_n_r = q_r[M_N] / rho_r
    var u_t1_r = q_r[M_T1] / rho_r
    var u_t2_r = q_r[M_T2] / rho_r
    var B_n_r = q_r[B_N]
    var B_t1_r = q_r[B_T1]
    var B_t2_r = q_r[B_T2]
    var p_T_r = p_gas_r + Real(0.5) * bsq_r
    var c_f_r = _fast_speed(rho_r, p_gas_r, B_n_r, bsq_r, gamma)

    var c_max = fmax(c_f_l, c_f_r)
    var S_L = (u_n_l if u_n_l < u_n_r else u_n_r) - c_max
    var S_R = (u_n_l if u_n_l > u_n_r else u_n_r) + c_max
    var B_n = Real(0.5) * (B_n_l + B_n_r)

    var udotB_l = u_n_l * B_n + u_t1_l * B_t1_l + u_t2_l * B_t2_l
    var udotB_r = u_n_r * B_n + u_t1_r * B_t1_r + u_t2_r * B_t2_r
    var F_L = Array[Real, MHD_NC](fill=Real(0.0))
    F_L[IDX_RHO] = rho_l * u_n_l
    F_L[M_N]  = rho_l * u_n_l * u_n_l + p_T_l - B_n * B_n
    F_L[M_T1] = rho_l * u_n_l * u_t1_l - B_n * B_t1_l
    F_L[M_T2] = rho_l * u_n_l * u_t2_l - B_n * B_t2_l
    F_L[IDX_E] = (q_l[IDX_E] + p_T_l) * u_n_l - B_n * udotB_l
    F_L[B_N]  = Real(0.0)
    F_L[B_T1] = B_t1_l * u_n_l - B_n * u_t1_l
    F_L[B_T2] = B_t2_l * u_n_l - B_n * u_t2_l
    var F_R = Array[Real, MHD_NC](fill=Real(0.0))
    F_R[IDX_RHO] = rho_r * u_n_r
    F_R[M_N]  = rho_r * u_n_r * u_n_r + p_T_r - B_n * B_n
    F_R[M_T1] = rho_r * u_n_r * u_t1_r - B_n * B_t1_r
    F_R[M_T2] = rho_r * u_n_r * u_t2_r - B_n * B_t2_r
    F_R[IDX_E] = (q_r[IDX_E] + p_T_r) * u_n_r - B_n * udotB_r
    F_R[B_N]  = Real(0.0)
    F_R[B_T1] = B_t1_r * u_n_r - B_n * u_t1_r
    F_R[B_T2] = B_t2_r * u_n_r - B_n * u_t2_r

    if S_L >= Real(0.0):
        return F_L^
    if S_R <= Real(0.0):
        return F_R^

    var U_L = Array[Real, MHD_NC](fill=Real(0.0))
    U_L[IDX_RHO] = rho_l
    U_L[M_N]  = q_l[M_N]
    U_L[M_T1] = q_l[M_T1]
    U_L[M_T2] = q_l[M_T2]
    U_L[IDX_E] = q_l[IDX_E]
    U_L[B_N]  = B_n
    U_L[B_T1] = B_t1_l
    U_L[B_T2] = B_t2_l
    var U_R = Array[Real, MHD_NC](fill=Real(0.0))
    U_R[IDX_RHO] = rho_r
    U_R[M_N]  = q_r[M_N]
    U_R[M_T1] = q_r[M_T1]
    U_R[M_T2] = q_r[M_T2]
    U_R[IDX_E] = q_r[IDX_E]
    U_R[B_N]  = B_n
    U_R[B_T1] = B_t1_r
    U_R[B_T2] = B_t2_r

    var sL_uL = S_L - u_n_l
    var sR_uR = S_R - u_n_r
    var num_sm = sR_uR * rho_r * u_n_r - sL_uL * rho_l * u_n_l - p_T_r + p_T_l
    var den_sm = sR_uR * rho_r - sL_uL * rho_l
    var S_M = num_sm / den_sm
    var p_T_star = p_T_l + rho_l * sL_uL * (S_M - u_n_l)

    var rho_L_s = rho_l * sL_uL / (S_L - S_M)
    var rho_R_s = rho_r * sR_uR / (S_R - S_M)
    var abs_Bn = fabs(B_n)
    var S_L_s = S_M - abs_Bn / sqrt(rho_L_s)
    var S_R_s = S_M + abs_Bn / sqrt(rho_R_s)
    var EPS = Real(1.0e-30)

    var den_L = rho_l * sL_uL * (S_L - S_M) - B_n * B_n
    var u_t1_L_s: Real
    var u_t2_L_s: Real
    var B_t1_L_s: Real
    var B_t2_L_s: Real
    if fabs(den_L) < EPS:
        u_t1_L_s = u_t1_l
        u_t2_L_s = u_t2_l
        B_t1_L_s = B_t1_l
        B_t2_L_s = B_t2_l
    else:
        var inv_den_L = Real(1.0) / den_L
        u_t1_L_s = u_t1_l - B_n * B_t1_l * (S_M - u_n_l) * inv_den_L
        u_t2_L_s = u_t2_l - B_n * B_t2_l * (S_M - u_n_l) * inv_den_L
        var num_BL = rho_l * sL_uL * sL_uL - B_n * B_n
        B_t1_L_s = B_t1_l * num_BL * inv_den_L
        B_t2_L_s = B_t2_l * num_BL * inv_den_L

    var den_R = rho_r * sR_uR * (S_R - S_M) - B_n * B_n
    var u_t1_R_s: Real
    var u_t2_R_s: Real
    var B_t1_R_s: Real
    var B_t2_R_s: Real
    if fabs(den_R) < EPS:
        u_t1_R_s = u_t1_r
        u_t2_R_s = u_t2_r
        B_t1_R_s = B_t1_r
        B_t2_R_s = B_t2_r
    else:
        var inv_den_R = Real(1.0) / den_R
        u_t1_R_s = u_t1_r - B_n * B_t1_r * (S_M - u_n_r) * inv_den_R
        u_t2_R_s = u_t2_r - B_n * B_t2_r * (S_M - u_n_r) * inv_den_R
        var num_BR = rho_r * sR_uR * sR_uR - B_n * B_n
        B_t1_R_s = B_t1_r * num_BR * inv_den_R
        B_t2_R_s = B_t2_r * num_BR * inv_den_R

    var udotB_L_s = S_M * B_n + u_t1_L_s * B_t1_L_s + u_t2_L_s * B_t2_L_s
    var udotB_R_s = S_M * B_n + u_t1_R_s * B_t1_R_s + u_t2_R_s * B_t2_R_s
    var E_L_s = (sL_uL * q_l[IDX_E] - p_T_l * u_n_l + p_T_star * S_M
                 + B_n * (udotB_l - udotB_L_s)) / (S_L - S_M)
    var E_R_s = (sR_uR * q_r[IDX_E] - p_T_r * u_n_r + p_T_star * S_M
                 + B_n * (udotB_r - udotB_R_s)) / (S_R - S_M)

    var U_L_s = Array[Real, MHD_NC](fill=Real(0.0))
    U_L_s[IDX_RHO] = rho_L_s
    U_L_s[M_N]  = rho_L_s * S_M
    U_L_s[M_T1] = rho_L_s * u_t1_L_s
    U_L_s[M_T2] = rho_L_s * u_t2_L_s
    U_L_s[IDX_E] = E_L_s
    U_L_s[B_N]  = B_n
    U_L_s[B_T1] = B_t1_L_s
    U_L_s[B_T2] = B_t2_L_s
    var U_R_s = Array[Real, MHD_NC](fill=Real(0.0))
    U_R_s[IDX_RHO] = rho_R_s
    U_R_s[M_N]  = rho_R_s * S_M
    U_R_s[M_T1] = rho_R_s * u_t1_R_s
    U_R_s[M_T2] = rho_R_s * u_t2_R_s
    U_R_s[IDX_E] = E_R_s
    U_R_s[B_N]  = B_n
    U_R_s[B_T1] = B_t1_R_s
    U_R_s[B_T2] = B_t2_R_s

    var F_L_s = Array[Real, MHD_NC](fill=Real(0.0))
    var F_R_s = Array[Real, MHD_NC](fill=Real(0.0))
    comptime for c in range(MHD_NC):
        F_L_s[c] = F_L[c] + S_L * (U_L_s[c] - U_L[c])
        F_R_s[c] = F_R[c] + S_R * (U_R_s[c] - U_R[c])

    if S_L_s >= Real(0.0):
        return F_L_s^
    if S_R_s <= Real(0.0):
        return F_R_s^

    var sqrt_rL = sqrt(rho_L_s)
    var sqrt_rR = sqrt(rho_R_s)
    var denom_ss = sqrt_rL + sqrt_rR
    var sign_Bn: Real = 1.0 if B_n >= Real(0.0) else -1.0
    var u_t1_ss = (sqrt_rL * u_t1_L_s + sqrt_rR * u_t1_R_s
                   + (B_t1_R_s - B_t1_L_s) * sign_Bn) / denom_ss
    var u_t2_ss = (sqrt_rL * u_t2_L_s + sqrt_rR * u_t2_R_s
                   + (B_t2_R_s - B_t2_L_s) * sign_Bn) / denom_ss
    var B_t1_ss = (sqrt_rL * B_t1_R_s + sqrt_rR * B_t1_L_s
                   + sqrt_rL * sqrt_rR * (u_t1_R_s - u_t1_L_s) * sign_Bn) / denom_ss
    var B_t2_ss = (sqrt_rL * B_t2_R_s + sqrt_rR * B_t2_L_s
                   + sqrt_rL * sqrt_rR * (u_t2_R_s - u_t2_L_s) * sign_Bn) / denom_ss
    var udotB_ss = S_M * B_n + u_t1_ss * B_t1_ss + u_t2_ss * B_t2_ss
    var E_L_ss = E_L_s - sqrt_rL * sign_Bn * (udotB_L_s - udotB_ss)
    var E_R_ss = E_R_s + sqrt_rR * sign_Bn * (udotB_R_s - udotB_ss)

    var U_L_ss = Array[Real, MHD_NC](fill=Real(0.0))
    U_L_ss[IDX_RHO] = rho_L_s
    U_L_ss[M_N]  = rho_L_s * S_M
    U_L_ss[M_T1] = rho_L_s * u_t1_ss
    U_L_ss[M_T2] = rho_L_s * u_t2_ss
    U_L_ss[IDX_E] = E_L_ss
    U_L_ss[B_N]  = B_n
    U_L_ss[B_T1] = B_t1_ss
    U_L_ss[B_T2] = B_t2_ss
    var U_R_ss = Array[Real, MHD_NC](fill=Real(0.0))
    U_R_ss[IDX_RHO] = rho_R_s
    U_R_ss[M_N]  = rho_R_s * S_M
    U_R_ss[M_T1] = rho_R_s * u_t1_ss
    U_R_ss[M_T2] = rho_R_s * u_t2_ss
    U_R_ss[IDX_E] = E_R_ss
    U_R_ss[B_N]  = B_n
    U_R_ss[B_T1] = B_t1_ss
    U_R_ss[B_T2] = B_t2_ss

    var F_chosen = Array[Real, MHD_NC](fill=Real(0.0))
    if S_M >= Real(0.0):
        comptime for c in range(MHD_NC):
            F_chosen[c] = F_L_s[c] + S_L_s * (U_L_ss[c] - U_L_s[c])
    else:
        comptime for c in range(MHD_NC):
            F_chosen[c] = F_R_s[c] + S_R_s * (U_R_ss[c] - U_R_s[c])
    return F_chosen^
