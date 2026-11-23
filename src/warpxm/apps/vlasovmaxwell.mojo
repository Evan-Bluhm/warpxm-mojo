# Kinetic Vlasov-Maxwell app.
#
# Mirrors upstream `src/apps/vlasovmaxwell/vlasovmaxwell.{h,cc}`. The
# unknown is the scalar distribution function `f(x, v, t)`; the
# accompanying Maxwell aux state `{E_x, E_y, E_z, B_x, B_y, B_z}` enters
# through the Lorentz acceleration `a = (q/m) (E + v × B) / λ_skin` when
# building the velocity-direction internal flux.
#
# `lorentz_acceleration_static[VDIM, axis]` gives that acceleration for
# one, two or three velocity dimensions; the terms of absent velocity
# components drop out at compile time (in 1V only `E_x` is left).
#
# Conserving the upstream parameter names:
#   * `charge_mass_ratio`  (q / m)
#   * `skin_depth_norm`    (λ_skin, normalising the Lorentz coupling)

from std.collections import Array

from warpxm.core.math_compat import fabs
from warpxm.core.types import Real


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
    # `vlasovmaxwell_calculate_field_component`.
    # ------------------------------------------------------------------

    @staticmethod
    def v_flux_factor_static(charge_mass_ratio: Real, skin_depth_norm: Real) -> Real:
        """Scalar coefficient `(q/m) / λ_skin` that multiplies the
        Lorentz `(E + v × B)` field when forming the v-axis flux."""
        return charge_mass_ratio / skin_depth_norm

    @staticmethod
    @always_inline
    def lorentz_acceleration_static[VDIM: Int, axis: Int](
        ex: Real, ey: Real, ez: Real,
        bx: Real, by: Real, bz: Real,
        v1: Real, v2: Real, v3: Real,
        factor: Real,
    ) -> Real:
        """Lorentz acceleration along velocity axis `axis` with `VDIM`
        velocity dimensions: `factor · (E + v × B)_axis`, `factor =
        (q/m) / λ_skin`. Velocity components at or beyond `VDIM` are zero
        and their terms are left out at compile time (a run-time
        `0.0 * B` is not folded away), so the `v` arguments for those
        components are ignored.

        The result never depends on the velocity component along `axis`
        itself, which is what makes the acceleration single-valued on a
        velocity face and constant along a velocity line of nodes."""
        comptime assert axis >= 0 and axis < VDIM, "axis must be below VDIM"
        var s: Real
        comptime match axis:
            case 0:
                # (v × B)_x = v2 B_z − v3 B_y
                s = ex
                comptime if VDIM >= 2:
                    s = s + v2 * bz
                comptime if VDIM >= 3:
                    s = s - v3 * by
            case 1:
                # (v × B)_y = v3 B_x − v1 B_z
                s = ey - v1 * bz
                comptime if VDIM >= 3:
                    s = s + v3 * bx
            case _:
                # (v × B)_z = v1 B_y − v2 B_x
                s = ez + v1 * by - v2 * bx
        return factor * s

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
        node."""
        var out = Array[Real, NC](fill=Real(0.0))
        comptime for c in range(NC):
            out[c] = av * f[c]
        return out^

    # ------------------------------------------------------------------
    # Numerical flux: upwind on the (face-local) signal speed. For
    # `F = c · f`, the upwind flux is
    #     F* = 0.5 (c · f_L + c · f_R) + 0.5 |c| (f_L − f_R).
    # The wave speed for a configuration face is the local v-coordinate;
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
