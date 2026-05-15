# Scalar linear advection in 3D.
#
# `∂q/∂t + ∇·(v q) = 0` with constant 3-vector velocity. Upwind numerical
# flux on a face with normal n = (nx, ny, nz). Same structure as
# `apps/advection_2d.mojo` lifted one dimension.

from std.collections import InlineArray

from warpxm.core.math_compat import fabs
from warpxm.core.types import Real


@fieldwise_init
struct Advection3D[NC: Int](ImplicitlyCopyable, Movable):
    """Scalar linear advection of `NC` independent tracer fields in 3D."""

    var velocity_x: Real
    var velocity_y: Real
    var velocity_z: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return Self.NC

    @staticmethod
    def numerical_flux_static(
        q_l: InlineArray[Real, Self.NC],
        q_r: InlineArray[Real, Self.NC],
        nx: Real, ny: Real, nz: Real,
        velocity_x: Real, velocity_y: Real, velocity_z: Real,
    ) -> InlineArray[Real, Self.NC]:
        var vn = velocity_x * nx + velocity_y * ny + velocity_z * nz
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

    @staticmethod
    def internal_flux_z_static(
        q: InlineArray[Real, Self.NC], velocity_z: Real,
    ) -> InlineArray[Real, Self.NC]:
        var f: InlineArray[Real, Self.NC] = InlineArray[Real, Self.NC](fill=Real(0.0))
        comptime for c in range(Self.NC):
            f[c] = velocity_z * q[c]
        return f
