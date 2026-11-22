# 5-moment Euler gas dynamics — single dimension-agnostic app.
#
# Mirrors the upstream `src/apps/5-moment/euler.{h,cc}` design: the app
# exposes axis-parametric internal-flux helpers and a single Rusanov
# numerical-flux function that takes a full 3-component unit normal.
# The DG launcher (`dg_core_{1,2,3}d.mojo`) owns the dimension — the
# app just operates on local node values.
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
# The numerical flux forms F·n directly for a 3-component unit normal
# (1D callers pass `nx = ±1, ny = 0, nz = 0`), one flux evaluation per
# side. It does not build f_x, f_y, f_z and dot them with n: IEEE
# arithmetic cannot fold `0.0 * f_y`, so that form pays for three fluxes
# per side on every face.
#
# Each state is divided by ρ exactly once (`inv_rho`); velocities and the
# kinetic energy are products with it. A double division costs roughly ten
# FP64 instructions on the device, which is the bottleneck on GPUs with a
# narrow FP64 pipe.

from std.collections import Array

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
def _rho_floored(q: Array[Real, EULER_NC], rho_min: Real) -> Real:
    return fmax(q[IDX_RHO], rho_min)


@always_inline
def _pressure_floored(
    q: Array[Real, EULER_NC], gamma: Real, rho_min: Real, press_min: Real,
) -> Real:
    """γ-law pressure with floors. Mirrors `euler_getPressureFloored`."""
    var inv_rho = Real(1.0) / _rho_floored(q, rho_min)
    return _pressure_from_inv_rho(q, inv_rho, gamma, press_min)


@always_inline
def _pressure_from_inv_rho(
    q: Array[Real, EULER_NC], inv_rho: Real, gamma: Real, press_min: Real,
) -> Real:
    var ke = Real(0.5) * (q[IDX_MX] * q[IDX_MX]
                          + q[IDX_MY] * q[IDX_MY]
                          + q[IDX_MZ] * q[IDX_MZ]) * inv_rho
    var p = (gamma - Real(1.0)) * (q[IDX_E] - ke)
    return fmax(p, press_min)


@always_inline
def _normal_flux_and_speed(
    q: Array[Real, EULER_NC],
    nx: Real, ny: Real, nz: Real,
    gamma: Real, rho_min: Real, press_min: Real,
) -> Array[Real, EULER_NC + 1]:
    """F(q)·n in slots 0..NC-1 and the signal speed |u·n| + c in slot NC."""
    var inv_rho = Real(1.0) / _rho_floored(q, rho_min)
    var p = _pressure_from_inv_rho(q, inv_rho, gamma, press_min)
    var m_n = q[IDX_MX] * nx + q[IDX_MY] * ny + q[IDX_MZ] * nz
    var u_n = m_n * inv_rho
    var out = Array[Real, EULER_NC + 1](fill=Real(0.0))
    out[IDX_RHO] = m_n
    out[IDX_MX] = q[IDX_MX] * u_n + p * nx
    out[IDX_MY] = q[IDX_MY] * u_n + p * ny
    out[IDX_MZ] = q[IDX_MZ] * u_n + p * nz
    out[IDX_E] = u_n * (q[IDX_E] + p)
    out[EULER_NC] = fabs(u_n) + sqrt(gamma * p * inv_rho)
    return out^


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
    @always_inline
    def internal_flux_axis_static[axis: Int](
        q: Array[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, EULER_NC]:
        comptime assert axis >= 0 and axis < 3, "axis must be 0, 1, or 2"
        var inv_rho = Real(1.0) / _rho_floored(q, rho_min)
        var p = _pressure_from_inv_rho(q, inv_rho, gamma, press_min)
        comptime mom_idx = IDX_MX + axis
        var u_axis = q[mom_idx] * inv_rho
        var f: Array[Real, EULER_NC] = Array[Real, EULER_NC](fill=Real(0.0))
        f[IDX_RHO] = q[mom_idx]
        f[IDX_MX] = q[IDX_MX] * u_axis
        f[IDX_MY] = q[IDX_MY] * u_axis
        f[IDX_MZ] = q[IDX_MZ] * u_axis
        f[mom_idx] += p
        f[IDX_E] = u_axis * (q[IDX_E] + p)
        return f^

    # Backward-compatible per-axis named methods. They forward to
    # `internal_flux_axis_static[axis]` so per-dim DG kernels can call
    # them directly without writing the axis parameter at every site.

    @staticmethod
    def internal_flux_x_static(
        q: Array[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, EULER_NC]:
        return Self.internal_flux_axis_static[0](q, gamma, rho_min, press_min)

    @staticmethod
    def internal_flux_y_static(
        q: Array[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, EULER_NC]:
        return Self.internal_flux_axis_static[1](q, gamma, rho_min, press_min)

    @staticmethod
    def internal_flux_z_static(
        q: Array[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, EULER_NC]:
        return Self.internal_flux_axis_static[2](q, gamma, rho_min, press_min)

    # The 1D-style entry point — always x-direction. Kept so the 1D DG
    # kernel doesn't need to spell `_x_static` for "the only axis I have".

    @staticmethod
    def internal_flux_static(
        q: Array[Real, EULER_NC],
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, EULER_NC]:
        return Self.internal_flux_axis_static[0](q, gamma, rho_min, press_min)

    # ------------------------------------------------------------------
    # Rusanov numerical flux on a 3-component unit normal.
    # 1D callers pass (±1, 0, 0); 2D callers pass (nx, ny, 0).
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: Array[Real, EULER_NC],
        q_r: Array[Real, EULER_NC],
        nx: Real, ny: Real, nz: Real,
        gamma: Real, rho_min: Real, press_min: Real,
    ) -> Array[Real, EULER_NC]:
        var fs_l = _normal_flux_and_speed(q_l, nx, ny, nz, gamma, rho_min, press_min)
        var fs_r = _normal_flux_and_speed(q_r, nx, ny, nz, gamma, rho_min, press_min)
        var c_max = fmax(fs_l[EULER_NC], fs_r[EULER_NC])

        var f: Array[Real, EULER_NC] = Array[Real, EULER_NC](fill=Real(0.0))
        comptime for c in range(EULER_NC):
            f[c] = Real(0.5) * (fs_l[c] + fs_r[c]) - Real(0.5) * c_max * (q_r[c] - q_l[c])
        return f^
