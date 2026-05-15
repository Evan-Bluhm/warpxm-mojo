# 5-moment Euler gas dynamics — single dimension-agnostic app.
#
# This is the consolidated version of the per-dimension Euler files
# (`euler_1d.mojo`, `euler_2d.mojo`, `euler_3d.mojo`). It mirrors the
# upstream `src/apps/5-moment/euler.{h,cc}` design: the app exposes axis-
# parametric internal-flux helpers and a single Rusanov numerical-flux
# function that takes a full 3-component unit normal. The DG launcher
# owns the dimension — the app just operates on local node values.
#
# Conserved variables (NC = 5):
#
#     q = (ρ, ρu_x, ρu_y, ρu_z, E)
#
# Pressure (γ-law EOS):
#
#     P = (γ - 1) · ( E - ½ · ρ · (u_x² + u_y² + u_z²) )
#
# Per-axis Euler flux tensors:
#
#     f_x = (ρu_x, ρu_x² + P, ρu_x u_y, ρu_x u_z, u_x (E + P))
#     f_y = (ρu_y, ρu_x u_y, ρu_y² + P, ρu_y u_z, u_y (E + P))
#     f_z = (ρu_z, ρu_x u_z, ρu_y u_z, ρu_z² + P, u_z (E + P))
#
# A 1D caller passes `nx = ±1, ny = 0, nz = 0`; the unused y/z terms in
# the numerical flux multiply by zero and the compiler folds them away.

from std.collections import InlineArray

from warpxm.core.math_compat import fabs, fmax, sqrt
from warpxm.core.types import Real


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
struct Euler(ImplicitlyCopyable, Movable):
    """5-moment Euler gas dynamics with Rusanov numerical flux."""

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
    # Axis-parametric internal flux. `axis = 0/1/2` selects x/y/z. The
    # DG launcher knows which axis it's integrating over.
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_axis_static[axis: Int](
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        comptime assert axis >= 0 and axis < 3, "axis must be 0, 1, or 2"
        var rho = _rho_floored(q, rho_min)
        var p = _pressure_floored(q, gamma, rho_min, press_min)
        comptime mom_idx = IDX_MX + axis
        var u_axis = q[mom_idx] / rho
        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        f[IDX_RHO] = q[mom_idx]
        f[IDX_MX] = q[IDX_MX] * u_axis
        f[IDX_MY] = q[IDX_MY] * u_axis
        f[IDX_MZ] = q[IDX_MZ] * u_axis
        f[mom_idx] += p
        f[IDX_E] = u_axis * (q[IDX_E] + p)
        return f

    # Backward-compatible per-axis named methods. They forward to
    # `internal_flux_axis_static[axis]` so per-dim DG kernels can call
    # them directly without writing the axis parameter at every site.

    @staticmethod
    def internal_flux_x_static(
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        return Self.internal_flux_axis_static[0](q, gamma, rho_min, press_min)

    @staticmethod
    def internal_flux_y_static(
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        return Self.internal_flux_axis_static[1](q, gamma, rho_min, press_min)

    @staticmethod
    def internal_flux_z_static(
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        return Self.internal_flux_axis_static[2](q, gamma, rho_min, press_min)

    # The 1D-style entry point. Pre-unification this was `internal_flux_static`
    # on `Euler1D` (always x-direction). Kept so the 1D DG kernel doesn't
    # need to spell `_x_static` to mean "the only axis I have".

    @staticmethod
    def internal_flux_static(
        q: InlineArray[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> InlineArray[Real, EULER_NC]:
        return Self.internal_flux_axis_static[0](q, gamma, rho_min, press_min)

    # ------------------------------------------------------------------
    # Rusanov numerical flux on a 3-component unit normal.
    # 1D callers pass (±1, 0, 0); 2D callers pass (nx, ny, 0).
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

        var fx_l = Self.internal_flux_axis_static[0](q_l, gamma, rho_min, press_min)
        var fy_l = Self.internal_flux_axis_static[1](q_l, gamma, rho_min, press_min)
        var fz_l = Self.internal_flux_axis_static[2](q_l, gamma, rho_min, press_min)
        var fx_r = Self.internal_flux_axis_static[0](q_r, gamma, rho_min, press_min)
        var fy_r = Self.internal_flux_axis_static[1](q_r, gamma, rho_min, press_min)
        var fz_r = Self.internal_flux_axis_static[2](q_r, gamma, rho_min, press_min)

        var f: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
        comptime for c in range(EULER_NC):
            var f_n_l = nx * fx_l[c] + ny * fy_l[c] + nz * fz_l[c]
            var f_n_r = nx * fx_r[c] + ny * fy_r[c] + nz * fz_r[c]
            f[c] = Real(0.5) * (f_n_l + f_n_r) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f
