# Maxwell's equations in 2D — linear electromagnetic wave propagation.
#
# 6 conserved components (E_x, E_y, E_z, B_x, B_y, B_z) carried so the full
# 3D Maxwell flux tensor lifts cleanly to 2D and 3D with the same code.
# Setting `∂/∂z = 0` in the 3D equations decouples the system into two
# independent modes:
#
#   TE mode (E_z, B_x, B_y): in-plane B, transverse E.
#   TM mode (B_z, E_x, E_y): in-plane E, transverse B.
#
# A 6-component formulation handles both at once with no per-mode branching.
#
# Equations of motion (no charges or currents, free-space):
#
#     ∂E/∂t = c² · ∇×B
#     ∂B/∂t = -∇×E
#
# In conservation-law form `∂q/∂t + ∂f_x/∂x + ∂f_y/∂y = 0`, the flux is
# linear in q:
#
#     f_x[E_x] = 0          f_y[E_x] = -c² · B_z
#     f_x[E_y] = c² · B_z   f_y[E_y] = 0
#     f_x[E_z] = -c² · B_y  f_y[E_z] = c² · B_x
#     f_x[B_x] = 0          f_y[B_x] = E_z
#     f_x[B_y] = -E_z       f_y[B_y] = 0
#     f_x[B_z] = E_y        f_y[B_z] = -E_x
#
# Numerical flux: central + Lax-Friedrichs dissipation. The max signal
# speed is `c` (the speed of light), uniform everywhere.

from std.collections import InlineArray

from warpxm.core.types import Real


comptime MAXWELL_NC: Int = 6

comptime IDX_EX: Int = 0
comptime IDX_EY: Int = 1
comptime IDX_EZ: Int = 2
comptime IDX_BX: Int = 3
comptime IDX_BY: Int = 4
comptime IDX_BZ: Int = 5


@fieldwise_init
struct Maxwell2D(ImplicitlyCopyable, Movable):
    """Linear Maxwell equations in 2D, parameterised by the wave speed `c`."""

    var c_light: Real
    """Speed of light. Normally `1.0` in dimensionless units."""

    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return MAXWELL_NC

    # ------------------------------------------------------------------
    # Internal-flux x-component. Linear in q.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal-flux y-component.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Numerical flux: central + Lax-Friedrichs dissipation with c_max = c.
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, MAXWELL_NC],
        q_r: InlineArray[Real, MAXWELL_NC],
        nx: Real,
        ny: Real,
        c: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        var fx_l = Self.internal_flux_x_static(q_l, c)
        var fy_l = Self.internal_flux_y_static(q_l, c)
        var fx_r = Self.internal_flux_x_static(q_r, c)
        var fy_r = Self.internal_flux_y_static(q_r, c)
        var f: InlineArray[Real, MAXWELL_NC] = InlineArray[Real, MAXWELL_NC](fill=Real(0.0))
        comptime for k in range(MAXWELL_NC):
            var f_n_l = nx * fx_l[k] + ny * fy_l[k]
            var f_n_r = nx * fx_r[k] + ny * fy_r[k]
            f[k] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c * (q_r[k] - q_l[k])
        return f
