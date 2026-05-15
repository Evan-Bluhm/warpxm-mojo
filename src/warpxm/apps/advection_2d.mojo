# Scalar linear advection in 2D.
#
# Mirrors `apps/advection.mojo` (1D) but with a 2-component velocity. The PDE
# is `∂q_c/∂t + ∇·(v q_c) = 0` for each component c, with v = (v_x, v_y).
# Each component is advected independently.
#
# Two complementary interfaces are provided:
#
# * Static methods taking scalar parameters (`velocity_x`, `velocity_y`) —
#   used inside GPU kernels, since structs holding the parameters would need
#   `DevicePassable` conformance.
# * Instance methods taking `self` — used in host-side glue / for the
#   `Advection1D[NC]` pattern.

from std.collections import InlineArray

from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, WXM_INF, WXM_TINY


@fieldwise_init
struct Advection2D[NC: Int](ImplicitlyCopyable, Movable):
    """Scalar linear advection of `NC` independent tracer fields in 2D."""

    var velocity_x: Real
    var velocity_y: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return Self.NC

    # ------------------------------------------------------------------
    # Static helpers — these are what the GPU kernel calls. They take all
    # physics parameters as plain scalars so the kernel does not need to
    # accept a `DevicePassable`-conforming struct.
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, Self.NC],
        q_r: InlineArray[Real, Self.NC],
        nx: Real,
        ny: Real,
        velocity_x: Real,
        velocity_y: Real,
    ) -> InlineArray[Real, Self.NC]:
        var vn = velocity_x * nx + velocity_y * ny
        var absc = fabs(vn)
        var flux: InlineArray[Real, Self.NC] = InlineArray[Real, Self.NC](fill=Real(0.0))
        comptime for c in range(Self.NC):
            flux[c] = Real(0.5) * (q_r[c] * (vn - absc) + q_l[c] * (vn + absc))
        return flux

    @staticmethod
    def internal_flux_x_static(
        q: InlineArray[Real, Self.NC], velocity_x: Real,
    ) -> InlineArray[Real, Self.NC]:
        var f: InlineArray[Real, Self.NC] = InlineArray[Real, Self.NC](fill=Real(0.0))
        comptime for c in range(Self.NC):
            f[c] = velocity_x * q[c]
        return f

    @staticmethod
    def internal_flux_y_static(
        q: InlineArray[Real, Self.NC], velocity_y: Real,
    ) -> InlineArray[Real, Self.NC]:
        var f: InlineArray[Real, Self.NC] = InlineArray[Real, Self.NC](fill=Real(0.0))
        comptime for c in range(Self.NC):
            f[c] = velocity_y * q[c]
        return f
