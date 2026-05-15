# 5-moment Euler gas dynamics in 1D.
#
# Same 5-component conservative formulation as `euler_2d.mojo` — keeping the
# (ρ, ρu_x, ρu_y, ρu_z, E) layout means 1D and 2D Euler share component
# indices and pressure routines. In 1D the y- and z-momenta are passive
# (they advect with the flow but never feel a transverse pressure gradient).
#
# Numerical flux: Rusanov (local Lax-Friedrichs).

from std.collections import InlineArray

from warpxm.apps.euler_2d import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.core.math_compat import fabs, fmax, sqrt
from warpxm.core.types import Real


@always_inline
def _rho_floored(q: InlineArray[Real, EULER_NC], rho_min: Real) -> Real:
    return fmax(q[IDX_RHO], rho_min)


@always_inline
def _pressure_floored(
    q: InlineArray[Real, EULER_NC], gamma: Real, rho_min: Real, press_min: Real,
) -> Real:
    var rho = _rho_floored(q, rho_min)
    var ke = Real(0.5) * (q[IDX_MX] * q[IDX_MX]
                          + q[IDX_MY] * q[IDX_MY]
                          + q[IDX_MZ] * q[IDX_MZ]) / rho
    var p = (gamma - Real(1.0)) * (q[IDX_E] - ke)
    return fmax(p, press_min)


@fieldwise_init
struct Euler1D(ImplicitlyCopyable, Movable):
    """5-moment Euler gas dynamics in 1D with Rusanov numerical flux."""

    var gas_gamma: Real
    var min_density: Real
    var min_pressure: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return EULER_NC

    # ------------------------------------------------------------------
    # Internal flux. In 1D only the x-direction matters:
    #     f_x = (ρu_x, ρu_x² + P, ρu_x u_y, ρu_x u_z, u_x (E + P))
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_static(
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

    # ------------------------------------------------------------------
    # Rusanov numerical flux. 1D face normal is `nx ∈ {+1, -1}` (or just +1
    # if we always evaluate fluxes in the +x direction, as we do here).
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, EULER_NC],
        q_r: InlineArray[Real, EULER_NC],
        nx: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        var rho_l = _rho_floored(q_l, rho_min)
        var rho_r = _rho_floored(q_r, rho_min)
        var p_l = _pressure_floored(q_l, gamma, rho_min, press_min)
        var p_r = _pressure_floored(q_r, gamma, rho_min, press_min)

        var u_n_l = (q_l[IDX_MX] * nx) / rho_l
        var u_n_r = (q_r[IDX_MX] * nx) / rho_r

        var c_l = sqrt(gamma * p_l / rho_l)
        var c_r = sqrt(gamma * p_r / rho_r)
        var c_max = fmax(fabs(u_n_l) + c_l, fabs(u_n_r) + c_r)

        var f_l = Self.internal_flux_static(q_l, gamma, rho_min, press_min)
        var f_r = Self.internal_flux_static(q_r, gamma, rho_min, press_min)

        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        comptime for c in range(EULER_NC):
            var f_n_l = nx * f_l[c]
            var f_n_r = nx * f_r[c]
            f[c] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f
