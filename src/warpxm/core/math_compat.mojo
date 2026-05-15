# Small math helpers callable from both host and device code.
#
# Mirrors the role of upstream's `wxm::math` in src/lib/wxm_math_compat.h. Most
# of upstream's helpers exist so the same code compiles for CPU and Kokkos
# device targets; here every helper is unconditionally device-callable because
# every kernel goes to the GPU.

from warpxm.core.types import Real

@always_inline
def fabs(x: Real) -> Real:
    return x if x >= 0.0 else -x

@always_inline
def sign(x: Real) -> Real:
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)

@always_inline
def fmin(a: Real, b: Real) -> Real:
    return a if a < b else b

@always_inline
def fmax(a: Real, b: Real) -> Real:
    return a if a > b else b

@always_inline
def square(x: Real) -> Real:
    return x * x

@always_inline
def sqrt(x: Real) -> Real:
    """Square root that works on host and on NVIDIA / AMD GPU device code.

    `std.math.sqrt` currently emits an approximate-sqrt intrinsic that's
    unsupported for `DType.float64` on NVIDIA GPUs. Routing through `x ** 0.5`
    falls back to the standard IEEE sqrt and lowers cleanly to both targets.
    """
    return x ** Real(0.5)
