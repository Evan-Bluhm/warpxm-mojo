# 5-moment Euler gas dynamics in 2D.
#
# Mirrors `src/apps/5-moment/euler.{h,cc}` from upstream WARPXM, restricted to
# 2D and to the Rusanov (local Lax-Friedrichs) numerical flux. Conserved
# variables are
#
#     q = (ρ, ρu_x, ρu_y, ρu_z, E)         (NC = 5)
#
# with a γ-law EOS:
#
#     P = (γ - 1) · ( E - ½ · ρ · (u_x² + u_y² + u_z²) )
#
# We keep `ρu_z` as a passive component — it gets advected with the flow but
# carries no z-momentum in pure 2D problems. This matches upstream's
# 5-component layout and makes the kernel trivial to lift to 2.5-D later.

from std.collections import InlineArray

from warpxm.core.math_compat import fabs, fmax, sqrt
from warpxm.core.types import Real, WXM_INF


comptime EULER_NC: Int = 5
"""Number of conserved components for the 5-moment Euler app."""

# Component indices.
comptime IDX_RHO: Int = 0
comptime IDX_MX: Int = 1
comptime IDX_MY: Int = 2
comptime IDX_MZ: Int = 3
comptime IDX_E: Int = 4


@always_inline
def _rho_floored(q: InlineArray[Real, EULER_NC], rho_min: Real) -> Real:
    return fmax(q[IDX_RHO], rho_min)


@always_inline
def _pressure_floored(
    q: InlineArray[Real, EULER_NC], gamma: Real, rho_min: Real, press_min: Real,
) -> Real:
    """γ-law pressure with floors. Mirrors `euler_getPressureFloored`."""
    var rho = _rho_floored(q, rho_min)
    var ke = Real(0.5) * (q[IDX_MX] * q[IDX_MX]
                          + q[IDX_MY] * q[IDX_MY]
                          + q[IDX_MZ] * q[IDX_MZ]) / rho
    var p = (gamma - Real(1.0)) * (q[IDX_E] - ke)
    return fmax(p, press_min)


@fieldwise_init
struct Euler2D(ImplicitlyCopyable, Movable):
    """5-moment Euler gas dynamics in 2D with Rusanov numerical flux."""

    var gas_gamma: Real
    """Ratio of specific heats (γ). Ideal monatomic: 5/3. Diatomic: 7/5."""

    var min_density: Real
    """Density floor used in numerical-flux pressure evaluation."""

    var min_pressure: Real
    """Pressure floor (clamped to ≥ this in flux computations)."""

    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return EULER_NC

    # ------------------------------------------------------------------
    # Static helpers — called from GPU kernels.
    #
    # Internal flux tensors:
    #
    #     f_x = (ρu_x, ρu_x² + P, ρu_x u_y, ρu_x u_z, u_x (E + P))
    #     f_y = (ρu_y, ρu_x u_y, ρu_y² + P, ρu_y u_z, u_y (E + P))
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_x_static(
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        var rho = _rho_floored(q, rho_min)
        var p = _pressure_floored(q, gamma, rho_min, press_min)
        var ux = q[IDX_MX] / rho
        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MX]
        f[IDX_MX] = q[IDX_MX] * ux + p
        f[IDX_MY] = q[IDX_MY] * ux
        f[IDX_MZ] = q[IDX_MZ] * ux
        f[IDX_E] = ux * (q[IDX_E] + p)
        return f

    @staticmethod
    def internal_flux_y_static(
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        var rho = _rho_floored(q, rho_min)
        var p = _pressure_floored(q, gamma, rho_min, press_min)
        var uy = q[IDX_MY] / rho
        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MY]
        f[IDX_MX] = q[IDX_MX] * uy
        f[IDX_MY] = q[IDX_MY] * uy + p
        f[IDX_MZ] = q[IDX_MZ] * uy
        f[IDX_E] = uy * (q[IDX_E] + p)
        return f

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, EULER_NC],
        q_r: InlineArray[Real, EULER_NC],
        nx: Real,
        ny: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        """Rusanov (local Lax-Friedrichs) numerical flux through a face with
        unit outward normal (nx, ny)."""
        var rho_l = _rho_floored(q_l, rho_min)
        var rho_r = _rho_floored(q_r, rho_min)
        var p_l = _pressure_floored(q_l, gamma, rho_min, press_min)
        var p_r = _pressure_floored(q_r, gamma, rho_min, press_min)

        var u_n_l = (q_l[IDX_MX] * nx + q_l[IDX_MY] * ny) / rho_l
        var u_n_r = (q_r[IDX_MX] * nx + q_r[IDX_MY] * ny) / rho_r

        var c_l = sqrt(gamma * p_l / rho_l)
        var c_r = sqrt(gamma * p_r / rho_r)
        var c_max = fmax(fabs(u_n_l) + c_l, fabs(u_n_r) + c_r)

        var fx_l = Self.internal_flux_x_static(q_l, gamma, rho_min, press_min)
        var fy_l = Self.internal_flux_y_static(q_l, gamma, rho_min, press_min)
        var fx_r = Self.internal_flux_x_static(q_r, gamma, rho_min, press_min)
        var fy_r = Self.internal_flux_y_static(q_r, gamma, rho_min, press_min)

        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        comptime for c in range(EULER_NC):
            var f_n_l = nx * fx_l[c] + ny * fy_l[c]
            var f_n_r = nx * fx_r[c] + ny * fy_r[c]
            f[c] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f
