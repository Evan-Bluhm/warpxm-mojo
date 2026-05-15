# 5-moment Euler gas dynamics in 3D.
#
# Same conservative layout as `apps/euler_2d.mojo` (NC = 5: ρ, ρu_x, ρu_y,
# ρu_z, E). The 3D version adds an `internal_flux_z_static` and a
# 3-component numerical flux that takes a unit normal (nx, ny, nz).
#
# Reuses the gas-pressure floors and the `EULER_NC` / index constants from
# `apps/euler_2d.mojo`.

from std.collections import InlineArray

from warpxm.apps.euler_2d import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    _rho_floored, _pressure_floored,
)
from warpxm.core.math_compat import fabs, fmax, sqrt
from warpxm.core.types import Real


@fieldwise_init
struct Euler3D(ImplicitlyCopyable, Movable):
    """5-moment Euler gas dynamics in 3D with Rusanov numerical flux."""

    var gas_gamma: Real
    var min_density: Real
    var min_pressure: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return EULER_NC

    # ------------------------------------------------------------------
    # Internal flux tensors:
    #
    #     f_x = (ρu_x, ρu_x² + P, ρu_x u_y, ρu_x u_z, u_x (E + P))
    #     f_y = (ρu_y, ρu_x u_y, ρu_y² + P, ρu_y u_z, u_y (E + P))
    #     f_z = (ρu_z, ρu_x u_z, ρu_y u_z, ρu_z² + P, u_z (E + P))
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
    def internal_flux_z_static(
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        var rho = _rho_floored(q, rho_min)
        var p = _pressure_floored(q, gamma, rho_min, press_min)
        var uz = q[IDX_MZ] / rho
        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        f[IDX_RHO] = q[IDX_MZ]
        f[IDX_MX] = q[IDX_MX] * uz
        f[IDX_MY] = q[IDX_MY] * uz
        f[IDX_MZ] = q[IDX_MZ] * uz + p
        f[IDX_E] = uz * (q[IDX_E] + p)
        return f

    # ------------------------------------------------------------------
    # Rusanov numerical flux on a 3D unit normal.
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, EULER_NC],
        q_r: InlineArray[Real, EULER_NC],
        nx: Real, ny: Real, nz: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        var rho_l = _rho_floored(q_l, rho_min)
        var rho_r = _rho_floored(q_r, rho_min)
        var p_l = _pressure_floored(q_l, gamma, rho_min, press_min)
        var p_r = _pressure_floored(q_r, gamma, rho_min, press_min)

        var u_n_l = (q_l[IDX_MX] * nx + q_l[IDX_MY] * ny + q_l[IDX_MZ] * nz) / rho_l
        var u_n_r = (q_r[IDX_MX] * nx + q_r[IDX_MY] * ny + q_r[IDX_MZ] * nz) / rho_r

        var c_l = sqrt(gamma * p_l / rho_l)
        var c_r = sqrt(gamma * p_r / rho_r)
        var c_max = fmax(fabs(u_n_l) + c_l, fabs(u_n_r) + c_r)

        var fx_l = Self.internal_flux_x_static(q_l, gamma, rho_min, press_min)
        var fy_l = Self.internal_flux_y_static(q_l, gamma, rho_min, press_min)
        var fz_l = Self.internal_flux_z_static(q_l, gamma, rho_min, press_min)
        var fx_r = Self.internal_flux_x_static(q_r, gamma, rho_min, press_min)
        var fy_r = Self.internal_flux_y_static(q_r, gamma, rho_min, press_min)
        var fz_r = Self.internal_flux_z_static(q_r, gamma, rho_min, press_min)

        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        comptime for c in range(EULER_NC):
            var f_n_l = nx * fx_l[c] + ny * fy_l[c] + nz * fz_l[c]
            var f_n_r = nx * fx_r[c] + ny * fy_r[c] + nz * fz_r[c]
            f[c] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f
