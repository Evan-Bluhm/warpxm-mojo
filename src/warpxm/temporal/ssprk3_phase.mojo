# SSPRK3 for a phase-space DG field, generic over the dimensions.
#
# `SSPRK3Phase[XDIM, VDIM, P_x, P_v, NC]` takes the same Shu-Osher stages
# as `SSPRK3[NDIM, P, NC]` (`temporal/ssprk3.mojo`) and has the same
# `step` / `run_until` / `run_until_cfl`, but keeps only two scratch
# copies of the state (one stage value and the right-hand side) instead
# of four. A phase-space state is the largest thing in a kinetic run, so
# the number of copies sets the largest mesh that fits on the device.
# The stages overwrite in place:
#
#     w   = u + dt L(u)
#     w   = 3/4 u + 1/4 (w + dt L(w))
#     u   = 1/3 u + 2/3 (w + dt L(w))

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major
from std.math import ceildiv

from warpxm.core.types import Real
from warpxm.spatial.dg_launch import RealTensor
from warpxm.temporal.pacing import CflPacer
from warpxm.temporal.ssprk3 import _euler_step_kernel
from warpxm.variable.phase_state import PhaseState


comptime _BLOCK: Int = 256


def _lincomb_inplace_kernel(
    dst: RealTensor,
    other: RealTensor,
    rhs: RealTensor,
    c_dst: Real,
    c_other: Real,
    c_rhs: Real,
    size_i64: Int64,
):
    """`dst[i] = c_dst · dst[i] + c_other · other[i] + c_rhs · rhs[i]`."""
    var t = Int(global_idx.x)
    if t < Int(size_i64):
        var d = rebind[Real](dst[t])
        var o = rebind[Real](other[t])
        var r = rebind[Real](rhs[t])
        dst[t] = rebind[dst.ElementType](c_dst * d + c_other * o + c_rhs * r)


def ssprk3_euler_stage(
    var ctx: DeviceContext,
    mut u: DeviceBuffer[Real.dtype],
    mut rhs: DeviceBuffer[Real.dtype],
    mut out: DeviceBuffer[Real.dtype],
    dt: Real,
    size: Int,
) raises:
    """`out = u + dt · rhs` over flat buffers of length `size`."""
    var layout = row_major(Int(size))
    comptime kernel = _euler_step_kernel[
        type_of(layout), type_of(layout), type_of(layout),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(u, layout), TileTensor(rhs, layout),
        TileTensor(out, layout),
        dt, Int64(size), grid_dim=ceildiv(size, _BLOCK), block_dim=_BLOCK,
    )


def ssprk3_inplace_stage(
    var ctx: DeviceContext,
    mut dst: DeviceBuffer[Real.dtype],
    mut other: DeviceBuffer[Real.dtype],
    mut rhs: DeviceBuffer[Real.dtype],
    c_dst: Real,
    c_other: Real,
    c_rhs: Real,
    size: Int,
) raises:
    """`dst = c_dst · dst + c_other · other + c_rhs · rhs` over flat
    buffers of length `size`, overwriting `dst`."""
    var layout = row_major(Int(size))
    ctx.enqueue_function[_lincomb_inplace_kernel](
        TileTensor(dst, layout), TileTensor(other, layout),
        TileTensor(rhs, layout),
        c_dst, c_other, c_rhs, Int64(size),
        grid_dim=ceildiv(size, _BLOCK), block_dim=_BLOCK,
    )


@always_inline
def no_op_adjust_phase[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int, NC: Int](
    mut q: PhaseState[XDIM, VDIM, P_x, P_v, NC],
) raises:
    """Identity adjuster: pass this when no adjuster step is wanted."""
    pass


struct SSPRK3Phase[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int, NC: Int]:
    """Three-stage Shu-Osher SSPRK3 stepper on a phase-space field."""

    comptime State = PhaseState[
        Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, Self.NC,
    ]

    var stage: Self.State
    var rhs: Self.State

    def __init__(
        out self, var ctx: DeviceContext, num_phase_elements: Int,
        num_ghost: Int = 0,
    ) raises:
        """Pass `mesh.num_phase_elements` and `mesh.num_ghost`."""
        self.stage = Self.State(ctx, num_phase_elements, num_ghost)
        self.rhs = Self.State(ctx, num_phase_elements, num_ghost)

    def step[
        ComputeRhs: def(mut Self.State, mut Self.State) raises -> None,
        Adjust: def(mut Self.State) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: Self.State,
        dt: Real,
        compute_rhs_fn: ComputeRhs,
        adjust_fn: Adjust,
    ) raises:
        var size = u.buf_len
        adjust_fn(u)

        # Stage 1: w = u^n + dt L(u^n)
        compute_rhs_fn(u, self.rhs)
        ssprk3_euler_stage(
            ctx, u.buf, self.rhs.buf, self.stage.buf, dt, size,
        )
        adjust_fn(self.stage)

        # Stage 2: w = 1/4 (w + dt L(w)) + 3/4 u^n
        compute_rhs_fn(self.stage, self.rhs)
        ssprk3_inplace_stage(
            ctx, self.stage.buf, u.buf, self.rhs.buf,
            Real(0.25), Real(0.75), Real(0.25) * dt, size,
        )
        adjust_fn(self.stage)

        # Stage 3: u^{n+1} = 1/3 u^n + 2/3 (w + dt L(w))
        compute_rhs_fn(self.stage, self.rhs)
        ssprk3_inplace_stage(
            ctx, u.buf, self.stage.buf, self.rhs.buf,
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0),
            Real(2.0) / Real(3.0) * dt, size,
        )
        adjust_fn(u)

    def run_until[
        ComputeRhs: def(mut Self.State, mut Self.State) raises -> None,
        Adjust: def(mut Self.State) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: Self.State,
        t: Real,
        t_target: Real,
        dt: Real,
        compute_rhs_fn: ComputeRhs,
        adjust_fn: Adjust,
    ) raises -> Real:
        """Advance from time `t` to `t_target` in steps of `dt`, shortening
        the last step to land on `t_target` exactly. Returns the time
        reached."""
        var time = t
        # The tolerance stops round-off in `time` from adding a ~1e-16 step.
        while time < t_target - Real(1.0e-9) * dt:
            var dt_use = dt if (time + dt) < t_target else (t_target - time)
            self.step(ctx, u, dt_use, compute_rhs_fn, adjust_fn)
            time += dt_use
        return time

    def run_until_cfl[
        ComputeRhs: def(mut Self.State, mut Self.State) raises -> None,
        Adjust: def(mut Self.State) raises -> None,
        SuggestDt: def(mut Self.State) raises -> Real,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: Self.State,
        t: Real,
        t_target: Real,
        compute_rhs_fn: ComputeRhs,
        adjust_fn: Adjust,
        suggest_dt_fn: SuggestDt,
        dt_interval: Int = 1,
        dt_min: Real = 0.0,
    ) raises -> Real:
        """Advance from `t` to `t_target` with the step chosen by
        `suggest_dt_fn(u)`, refreshed every `dt_interval` steps. Raises if
        the suggestion is not above `dt_min`. Returns the time reached."""
        var time = t
        var pacer = CflPacer(dt_interval, dt_min)
        while time < t_target:
            if pacer.due():
                pacer.refresh(suggest_dt_fn(u), time)
            var remaining = t_target - time
            var dt_use = pacer.next(remaining)
            self.step(ctx, u, dt_use, compute_rhs_fn, adjust_fn)
            if dt_use == remaining:
                return t_target
            time += dt_use
        return time
