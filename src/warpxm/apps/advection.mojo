# Linear advection app — single dimension-agnostic, NC-parametric version.
#
# Mirrors `src/apps/simple/advection.{h,cc}` upstream. The PDE is
#
#     ∂q_c/∂t + ∇·(v q_c) = 0      for each component c ∈ 0..NC-1
#
# with a single constant 3-vector velocity. Each component evolves
# independently. The DG core supplies the face
# normal; the app dots velocity into the normal and applies the upwind
# formula. 1D callers store `velocity_x` (with `velocity_y = velocity_z =
# 0` defaults); 2D and 3D add the extra components.

from std.collections import Array

from warpxm.core.math_compat import fabs
from warpxm.core.types import Real


struct Advection[NC: Int](ImplicitlyCopyable, Movable):
    """Linear advection of `NC` independent tracer fields."""

    var velocity_x: Real
    var velocity_y: Real
    var velocity_z: Real
    var cfl_max: Real

    def __init__(
        out self,
        velocity_x: Real = Real(0.0),
        velocity_y: Real = Real(0.0),
        velocity_z: Real = Real(0.0),
        cfl_max: Real = Real(1.0),
    ):
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z
        self.cfl_max = cfl_max

    @staticmethod
    def num_components() -> Int:
        return Self.NC

    # ------------------------------------------------------------------
    # Axis-parametric internal flux. `axis = 0/1/2` selects the velocity
    # component to scale by.
    # ------------------------------------------------------------------

    @staticmethod
    def internal_flux_axis_static[axis: Int](
        q: Array[Real, Self.NC],
        velocity_x: Real, velocity_y: Real, velocity_z: Real,
    ) -> Array[Real, Self.NC]:
        comptime assert axis >= 0 and axis < 3, "axis must be 0, 1, or 2"
        var v: Real = Real(0.0)
        comptime if axis == 0:
            v = velocity_x
        comptime if axis == 1:
            v = velocity_y
        comptime if axis == 2:
            v = velocity_z
        var f: Array[Real, Self.NC] = Array[Real, Self.NC](fill=Real(0.0))
        comptime for c in range(Self.NC):
            f[c] = v * q[c]
        return f^

    @staticmethod
    def internal_flux_x_static(
        q: Array[Real, Self.NC], velocity_x: Real,
    ) -> Array[Real, Self.NC]:
        # Single-axis convenience form — only the x-velocity matters here.
        return Self.internal_flux_axis_static[0](q, velocity_x, Real(0.0), Real(0.0))

    @staticmethod
    def internal_flux_y_static(
        q: Array[Real, Self.NC], velocity_y: Real,
    ) -> Array[Real, Self.NC]:
        return Self.internal_flux_axis_static[1](q, Real(0.0), velocity_y, Real(0.0))

    @staticmethod
    def internal_flux_z_static(
        q: Array[Real, Self.NC], velocity_z: Real,
    ) -> Array[Real, Self.NC]:
        return Self.internal_flux_axis_static[2](q, Real(0.0), Real(0.0), velocity_z)

    # ------------------------------------------------------------------
    # Upwind numerical flux on a 3-component unit normal.
    # 1D callers pass (±1, 0, 0); 2D callers pass (nx, ny, 0).
    # ------------------------------------------------------------------

    @staticmethod
    def numerical_flux_static(
        q_l: Array[Real, Self.NC],
        q_r: Array[Real, Self.NC],
        nx: Real, ny: Real, nz: Real,
        velocity_x: Real, velocity_y: Real, velocity_z: Real,
    ) -> Array[Real, Self.NC]:
        var vn = velocity_x * nx + velocity_y * ny + velocity_z * nz
        var absc = fabs(vn)
        var flux: Array[Real, Self.NC] = Array[Real, Self.NC](fill=Real(0.0))
        comptime for c in range(Self.NC):
            flux[c] = Real(0.5) * (q_r[c] * (vn - absc) + q_l[c] * (vn + absc))
        return flux^
