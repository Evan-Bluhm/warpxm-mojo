# Kinetic Vlasov-Maxwell app.
#
# Mirrors upstream `src/apps/vlasovmaxwell/vlasovmaxwell.{h,cc}`. The
# unknown is the scalar distribution function `f(x, v, t)`; the
# accompanying Maxwell aux state `{E_x, E_y, E_z, B_x, B_y, B_z}` enters
# through the Lorentz acceleration `a = (q/m) (E + v × B) / λ_skin` when
# building the velocity-direction internal flux.
#
# In 1D1V (one physical dim, one velocity dim) the Lorentz formula
# collapses: `v_y = v_z = 0`, so the velocity-direction field component
# is simply `(q/m)/λ_skin · E_x`. The magnetic field drops out entirely.
# 2D2V / 3D3V ports will need the full `E + v × B` expression and
# additional internal-flux methods along the extra velocity axes.
#
# Conserving the upstream parameter names:
#   * `charge_mass_ratio`  (q / m)
#   * `skin_depth_norm`    (λ_skin, normalising the Lorentz coupling)

from std.collections import Array

from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, WXM_INF, WXM_TINY


# Number of solution components on the Vlasov side: just `f`.
comptime VLASOV_NC: Int = 1

# Number of components on the Maxwell aux side: {E_x, E_y, E_z, B_x, B_y, B_z}.
comptime VLASOV_AUX_NC: Int = 6

# Maxwell aux indices, parallel to `apps/maxwell.mojo`.
comptime AUX_EX: Int = 0
comptime AUX_EY: Int = 1
comptime AUX_EZ: Int = 2
comptime AUX_BX: Int = 3
comptime AUX_BY: Int = 4
comptime AUX_BZ: Int = 5


struct VlasovMaxwell(ImplicitlyCopyable, Movable):
    """Single-species kinetic Vlasov-Maxwell app constants.

    `charge_mass_ratio` and `skin_depth_norm` match the upstream
    `constants_vlasovmaxwell_t` fields (see
    `~/GitHub/warpxm/src/apps/vlasovmaxwell/vlasovmaxwell.h`)."""

    var charge_mass_ratio: Real
    """`q / m` for the species. Negative for electrons."""

    var skin_depth_norm: Real
    """Skin-depth normalisation `λ_skin` dividing the Lorentz coupling.
    Together with `charge_mass_ratio` it forms the velocity-flux factor
    `(q/m) / λ_skin` that multiplies `(E + v × B)`."""

    var cfl_max: Real
    """Upper bound on the CFL number, applied to whichever phase-space
    direction binds first (the v-advection terms or the E+v×B
    acceleration terms — upstream uses a single combined `Σ |v_d / dx_d|
    + |a_d / dv_d|` summation)."""

    def __init__(
        out self,
        charge_mass_ratio: Real = Real(-1.0),
        skin_depth_norm: Real = Real(1.0),
        cfl_max: Real = Real(0.5),
    ):
        self.charge_mass_ratio = charge_mass_ratio
        self.skin_depth_norm = skin_depth_norm
        self.cfl_max = cfl_max

    @staticmethod
    def num_components() -> Int:
        return VLASOV_NC

    @staticmethod
    def num_aux_components() -> Int:
        return VLASOV_AUX_NC

    # ------------------------------------------------------------------
    # Per-node helpers. Mirror upstream's
    # `vlasovmaxwell_calculate_field_component` collapsed to 1D1V (where
    # v_y = v_z = 0 and only E_x contributes to the v-axis acceleration).
    # ------------------------------------------------------------------

    @staticmethod
    def v_flux_factor_static(charge_mass_ratio: Real, skin_depth_norm: Real) -> Real:
        """Scalar coefficient `(q/m) / λ_skin` that multiplies the
        Lorentz `(E + v × B)` field when forming the v-axis flux."""
        return charge_mass_ratio / skin_depth_norm

    @staticmethod
    def acceleration_v_1d1v_static(
        ex: Real, charge_mass_ratio: Real, skin_depth_norm: Real,
    ) -> Real:
        """1D1V Lorentz acceleration along the single v-axis:
        `a_v = (q/m)/λ_skin · E_x`. `v × B` vanishes because the only
        velocity component is `v_x` and it crosses with no other axis."""
        return (charge_mass_ratio / skin_depth_norm) * ex

    @staticmethod
    def acceleration_v1_1d2v_static(
        ex: Real, bz: Real, v2: Real,
        charge_mass_ratio: Real, skin_depth_norm: Real,
    ) -> Real:
        """1D2V Lorentz acceleration along `v1` (= `v_x`):
        `a_v1 = (q/m)/λ_skin · (E_x + v2·B_z)`.

        Derivation: with `B = (0, 0, B_z)` and `v = (v1, v2, 0)`,
        `(v × B)_x = v2·B_z`. The `v3·B_y` term that would also appear in
        a full 3V solver collapses because `v3 = 0` in 1D2V."""
        return (charge_mass_ratio / skin_depth_norm) * (ex + v2 * bz)

    @staticmethod
    def acceleration_v2_1d2v_static(
        ey: Real, bz: Real, v1: Real,
        charge_mass_ratio: Real, skin_depth_norm: Real,
    ) -> Real:
        """1D2V Lorentz acceleration along `v2` (= `v_y`):
        `a_v2 = (q/m)/λ_skin · (E_y − v1·B_z)`.

        Derivation: `(v × B)_y = -v1·B_z` with the same `(B, v)` from
        `acceleration_v1_1d2v_static`. Together with `a_v1`, this is the
        full gyration-driving piece of the 1D2V Lorentz force."""
        return (charge_mass_ratio / skin_depth_norm) * (ey - v1 * bz)

    # ------------------------------------------------------------------
    # Internal fluxes — linear in `f`, so they're just `c · f` with `c`
    # the per-node signal speed (v-coordinate for the x-axis, Lorentz
    # acceleration for the v-axis). Returned as `Array[Real, NC]`
    # to keep the same shape as the other apps' flux helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_x_static[NC: Int](
        f: Array[Real, NC], vx: Real,
    ) -> Array[Real, NC]:
        """Physical-x flux: `vx · f` per component. `vx` is the
        v-coordinate of the current phase node."""
        var out = Array[Real, NC](fill=Real(0.0))
        comptime for c in range(NC):
            out[c] = vx * f[c]
        return out^

    @staticmethod
    def internal_flux_v_static[NC: Int](
        f: Array[Real, NC], av: Real,
    ) -> Array[Real, NC]:
        """Velocity-axis flux: `a_v · f` per component. `av` is the
        Lorentz acceleration along the v-axis at the current phase
        node — e.g. `acceleration_v_1d1v_static(Ex, q/m, λ)`."""
        var out = Array[Real, NC](fill=Real(0.0))
        comptime for c in range(NC):
            out[c] = av * f[c]
        return out^

    @staticmethod
    def internal_flux_v1_static[NC: Int](
        f: Array[Real, NC], av1: Real,
    ) -> Array[Real, NC]:
        """1D2V v1-axis flux: `a_v1 · f` per component. Thin alias over
        `internal_flux_v_static`; named separately so the 1D2V DG
        kernel's call site reads symmetrically with `internal_flux_x_static`
        and `internal_flux_v2_static`."""
        return Self.internal_flux_v_static[NC](f, av1)

    @staticmethod
    def internal_flux_v2_static[NC: Int](
        f: Array[Real, NC], av2: Real,
    ) -> Array[Real, NC]:
        """1D2V v2-axis flux: `a_v2 · f` per component. Counterpart to
        `internal_flux_v1_static`."""
        return Self.internal_flux_v_static[NC](f, av2)

    # ------------------------------------------------------------------
    # Numerical flux: upwind on the (face-local) signal speed. For
    # `F = c · f`, the upwind flux is
    #     F* = 0.5 (c · f_L + c · f_R) + 0.5 |c| (f_L − f_R).
    # In 1D1V the wave speed for an x-axis face is the local v-coordinate;
    # for a v-axis face it is the local Lorentz acceleration. The caller
    # passes whichever applies via `wave_speed`.
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static[NC: Int](
        f_l: Array[Real, NC], f_r: Array[Real, NC],
        wave_speed: Real,
    ) -> Array[Real, NC]:
        """Upwind flux on a single phase-space face. `wave_speed > 0`
        ⇒ pulls from the left state; `wave_speed < 0` ⇒ from the
        right. Identical formula on both axes — the caller decides
        which speed to feed in."""
        var abs_c = fabs(wave_speed)
        var out = Array[Real, NC](fill=Real(0.0))
        comptime for c in range(NC):
            out[c] = Real(0.5) * (
                wave_speed * (f_l[c] + f_r[c]) + abs_c * (f_l[c] - f_r[c])
            )
        return out^

    # ------------------------------------------------------------------
    # CFL helper. Mirrors upstream's combined-bound formula in
    # `vlasovmaxwell_internal_flux`: dt_max = cfl / Σ |c_d| / Δ_d, where
    # the sum is over all phase-space directions.
    # ------------------------------------------------------------------

    def dt_max_1d1v(self, vx: Real, av: Real, dx: Real, dv: Real) -> Real:
        """Per-node CFL bound on `dt`. Caller passes the local
        v-coordinate (`vx`) and local Lorentz acceleration (`av`)
        together with mesh widths. Returns `WXM_INF` if both speeds
        underflow `WXM_TINY`."""
        var s = fabs(vx) / dx + fabs(av) / dv
        if s < WXM_TINY:
            return WXM_INF
        return self.cfl_max / s
