# Scalar linear advection app.
#
# Mirrors `src/apps/simple/advection.{h,cc}` from upstream WARPXM. The PDE is
#
#     ∂q_c/∂t + ∇·(v q_c) = 0   for each component c ∈ 0..NC-1
#
# with a single constant advection velocity `v` shared by all components.
# Each component evolves independently — no coupling — which is the simplest
# case to use as a multi-component vertical slice.
#
# In the GPU-only world we don't need the kernel-launcher registry plumbing —
# the spatial solver is templated on the app type and inlines the flux
# functions directly.

from std.collections import InlineArray

from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, WXM_INF, WXM_TINY


@fieldwise_init
struct Advection1D[NC: Int](ImplicitlyCopyable, Movable):
    """Scalar linear advection of `NC` independent tracer fields in 1D."""

    var velocity: Real
    var cfl_max: Real

    @staticmethod
    def num_components() -> Int:
        return Self.NC

    @staticmethod
    def numerical_flux(
        q_l: InlineArray[Real, Self.NC],
        q_r: InlineArray[Real, Self.NC],
        normal: Real,
        velocity: Real,
    ) -> InlineArray[Real, Self.NC]:
        """Upwind numerical flux, applied component-wise.

        Returns a length-NC vector of fluxes on the face in the direction of
        `normal`. Each component is decoupled, so the same upwind formula
        applies to every component.
        """
        var nc = velocity * normal
        var absc = fabs(nc)
        var flux: InlineArray[Real, Self.NC] = InlineArray[Real, Self.NC](fill=0.0)
        comptime for c in range(Self.NC):
            flux[c] = Real(0.5) * (q_r[c] * (nc - absc) + q_l[c] * (nc + absc))
        return flux

    @staticmethod
    def internal_flux(
        q: InlineArray[Real, Self.NC], velocity: Real
    ) -> InlineArray[Real, Self.NC]:
        """Linear internal flux `f_c(q) = v · q_c`, component-wise."""
        var flux: InlineArray[Real, Self.NC] = InlineArray[Real, Self.NC](fill=0.0)
        comptime for c in range(Self.NC):
            flux[c] = velocity * q[c]
        return flux

    @staticmethod
    def suggested_dt(normal: Real, dxn: Real, cfl_max: Real, velocity: Real) -> Real:
        """CFL-limited timestep contribution from one face."""
        var nc = velocity * normal
        var absc = fabs(nc)
        return (cfl_max * dxn / absc) if absc > WXM_TINY else WXM_INF
