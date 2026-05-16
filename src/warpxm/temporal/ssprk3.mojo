# Device-resident SSPRK3 (Shu-Osher) explicit Runge-Kutta time integrator.
#
# Three stages on the standard SSPRK3 Butcher form:
#
#     u^(1)   = u^n   + Δt · L(u^n)
#     u^(2)   = 3/4 · u^n   + 1/4 · ( u^(1) + Δt · L(u^(1)) )
#     u^{n+1} = 1/3 · u^n   + 2/3 · ( u^(2) + Δt · L(u^(2)) )
#
# All scratch arrays live on the device. Time-step selection is left to the
# caller for now — we accept Δt as an argument. That mirrors the early
# `warpy.dt_calc.fixed_dt` path from upstream; a CFL-driven controller can be
# layered on later.

from std.gpu import global_idx
from layout import TileTensor, TensorLayout

from warpxm.core.types import Real, WXM_DTYPE


# ---------------------------------------------------------------------------
# Kernel: output[i] = x[i] + dt · rhs[i]
#
# Fused forward-Euler stage — only two inputs, written to a distinct output.
# Used by SSPRK3 stage 1 (where the "u_stage" and "u0" of the more general
# combine would alias each other and fall foul of Mojo's aliasing checker).
# ---------------------------------------------------------------------------

def _euler_step_kernel[
    XLayout: TensorLayout, RhsLayout: TensorLayout, OutLayout: TensorLayout,
](
    x: TileTensor[WXM_DTYPE, XLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    output: TileTensor[WXM_DTYPE, OutLayout, MutAnyOrigin],
    dt: Real,
    size: Int,
):
    comptime assert x.flat_rank == 1, "euler_step expects 1D x"
    comptime assert rhs.flat_rank == 1, "euler_step expects 1D rhs"
    comptime assert output.flat_rank == 1, "euler_step expects 1D output"
    var t = global_idx.x
    if t < size:
        var xv = rebind[Scalar[WXM_DTYPE]](x[t])
        var rv = rebind[Scalar[WXM_DTYPE]](rhs[t])
        output[t] = rebind[output.ElementType](xv + dt * rv)


# ---------------------------------------------------------------------------
# Kernel: out[i] = a · u0[i] + b · ( u_stage[i] + dt · rhs[i] )
#
# Used by the SSPRK3 combination steps. `out` is allowed to alias `u_stage`.
# ---------------------------------------------------------------------------

def _ssprk3_combine_kernel[
    U0Layout: TensorLayout,
    USLayout: TensorLayout,
    RhsLayout: TensorLayout,
    OutLayout: TensorLayout,
](
    u0: TileTensor[WXM_DTYPE, U0Layout, MutAnyOrigin],
    u_stage: TileTensor[WXM_DTYPE, USLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    output: TileTensor[WXM_DTYPE, OutLayout, MutAnyOrigin],
    a: Real,
    b: Real,
    dt: Real,
    size: Int,
):
    comptime assert u0.flat_rank == 1, "combine expects 1D u0"
    comptime assert u_stage.flat_rank == 1, "combine expects 1D u_stage"
    comptime assert rhs.flat_rank == 1, "combine expects 1D rhs"
    comptime assert output.flat_rank == 1, "combine expects 1D output"
    var t = global_idx.x
    if t < size:
        var u0v = rebind[Scalar[WXM_DTYPE]](u0[t])
        var usv = rebind[Scalar[WXM_DTYPE]](u_stage[t])
        var rhv = rebind[Scalar[WXM_DTYPE]](rhs[t])
        output[t] = rebind[output.ElementType](a * u0v + b * (usv + dt * rhv))
