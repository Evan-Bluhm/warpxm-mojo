# Small math helpers callable from both host and device code.
#
# Mirrors the role of upstream's `wxm::math` in src/lib/wxm_math_compat.h. Most
# of upstream's helpers exist so the same code compiles for CPU and Kokkos
# device targets; here every helper is unconditionally device-callable because
# every kernel goes to the GPU.

from std.sys import llvm_intrinsic

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
    """IEEE square root on host and device, via the `llvm.sqrt` intrinsic.

    On NVIDIA GPUs this lowers to the native `sqrt.rn.f64` instruction.
    Do not write `x ** 0.5` instead: that calls the general `pow` routine,
    which is many times more expensive in a kernel (it was the largest
    single cost in the MHD flux before this was changed).
    """
    return llvm_intrinsic["llvm.sqrt", Real, has_side_effect=False](x)
