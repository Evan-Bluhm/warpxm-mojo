# Maxwell's equations in 3D — linear electromagnetic wave propagation.
#
# Same 6-component conservative layout as `apps/maxwell_2d.mojo`. In 3D the
# flux tensor has non-zero components on all three axes; the 2D version
# silently dropped the z-axis pieces (because ∂/∂z = 0 there).
#
# Equations: ∂E/∂t = c² ∇×B,  ∂B/∂t = −∇×E.
#
# Componentwise flux divergences `∂q/∂t + ∂f_x/∂x + ∂f_y/∂y + ∂f_z/∂z = 0`:
#
#     f_x[E_x]  = 0          f_y[E_x] = −c²·B_z   f_z[E_x] = +c²·B_y
#     f_x[E_y]  = +c²·B_z    f_y[E_y] = 0         f_z[E_y] = −c²·B_x
#     f_x[E_z]  = −c²·B_y    f_y[E_z] = +c²·B_x   f_z[E_z] = 0
#     f_x[B_x]  = 0          f_y[B_x] = E_z       f_z[B_x] = −E_y
#     f_x[B_y]  = −E_z       f_y[B_y] = 0         f_z[B_y] = +E_x
#     f_x[B_z]  = E_y        f_y[B_z] = −E_x      f_z[B_z] = 0
#
# Numerical flux: central + Lax-Friedrichs dissipation at signal speed c.

from std.collections import InlineArray

from warpxm.apps.maxwell_2d import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.types import Real


@fieldwise_init
struct Maxwell3D(ImplicitlyCopyable, Movable):
    """Linear Maxwell equations in 3D, parameterised by the wave speed `c`."""

    var c_light: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return MAXWELL_NC

    @staticmethod
    def internal_flux_x_static(
        q: InlineArray[Real, MAXWELL_NC], c: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        var c2 = c * c
        var f: InlineArray[Real, MAXWELL_NC] = InlineArray[Real, MAXWELL_NC](fill=Real(0.0))
        f[IDX_EX] = Real(0.0)
        f[IDX_EY] = c2 * q[IDX_BZ]
        f[IDX_EZ] = -c2 * q[IDX_BY]
        f[IDX_BX] = Real(0.0)
        f[IDX_BY] = -q[IDX_EZ]
        f[IDX_BZ] = q[IDX_EY]
        return f

    @staticmethod
    def internal_flux_y_static(
        q: InlineArray[Real, MAXWELL_NC], c: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        var c2 = c * c
        var f: InlineArray[Real, MAXWELL_NC] = InlineArray[Real, MAXWELL_NC](fill=Real(0.0))
        f[IDX_EX] = -c2 * q[IDX_BZ]
        f[IDX_EY] = Real(0.0)
        f[IDX_EZ] = c2 * q[IDX_BX]
        f[IDX_BX] = q[IDX_EZ]
        f[IDX_BY] = Real(0.0)
        f[IDX_BZ] = -q[IDX_EX]
        return f

    @staticmethod
    def internal_flux_z_static(
        q: InlineArray[Real, MAXWELL_NC], c: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        var c2 = c * c
        var f: InlineArray[Real, MAXWELL_NC] = InlineArray[Real, MAXWELL_NC](fill=Real(0.0))
        f[IDX_EX] = c2 * q[IDX_BY]
        f[IDX_EY] = -c2 * q[IDX_BX]
        f[IDX_EZ] = Real(0.0)
        f[IDX_BX] = -q[IDX_EY]
        f[IDX_BY] = q[IDX_EX]
        f[IDX_BZ] = Real(0.0)
        return f

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, MAXWELL_NC],
        q_r: InlineArray[Real, MAXWELL_NC],
        nx: Real, ny: Real, nz: Real,
        c: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        var fx_l = Self.internal_flux_x_static(q_l, c)
        var fy_l = Self.internal_flux_y_static(q_l, c)
        var fz_l = Self.internal_flux_z_static(q_l, c)
        var fx_r = Self.internal_flux_x_static(q_r, c)
        var fy_r = Self.internal_flux_y_static(q_r, c)
        var fz_r = Self.internal_flux_z_static(q_r, c)
        var f: InlineArray[Real, MAXWELL_NC] = InlineArray[Real, MAXWELL_NC](fill=Real(0.0))
        comptime for k in range(MAXWELL_NC):
            var f_n_l = nx * fx_l[k] + ny * fy_l[k] + nz * fz_l[k]
            var f_n_r = nx * fx_r[k] + ny * fy_r[k] + nz * fz_r[k]
            f[k] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c * (q_r[k] - q_l[k])
        return f
