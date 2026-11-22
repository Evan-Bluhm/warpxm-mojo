# Generic SSPRK3 stepper for 1D1V phase-space DG fields.
#
# Mirrors `temporal/ssprk3_{1,2,3}d.mojo` but parameterised by separate
# physical-space and velocity-space orders `(P_x, P_v)` and operating on
# `PhaseState1D1V` buffers. The combine kernels themselves
# (`_euler_step_kernel`, `_ssprk3_combine_kernel` in `ssprk3.mojo`) are
# flat-buffer ops and require no change — this stepper just hands them
# the phase-state buffers.

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.math import ceildiv

from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.phase_state_1d1v import PhaseState1D1V


@always_inline
def no_op_adjust_1d1v[P_x: Int, P_v: Int, NC: Int](
    mut q: PhaseState1D1V[P_x, P_v, NC],
) raises:
    """Identity adjuster — pass this when no adjuster step is wanted.
    The Vlasov-Maxwell port does not currently ship a phase-space
    limiter, so this is the only adjuster every driver needs for now."""
    pass


struct SSPRK3_1D1V[P_x: Int, P_v: Int, NC: Int]:
    """Three-stage Shu-Osher SSPRK3 stepper on a 1D1V phase-space field."""

    var u_save: PhaseState1D1V[Self.P_x, Self.P_v, Self.NC]
    var stage1: PhaseState1D1V[Self.P_x, Self.P_v, Self.NC]
    var stage2: PhaseState1D1V[Self.P_x, Self.P_v, Self.NC]
    var rhs: PhaseState1D1V[Self.P_x, Self.P_v, Self.NC]

    def __init__(
        out self, var ctx: DeviceContext, num_x: Int, num_v: Int,
    ) raises:
        self.u_save = PhaseState1D1V[Self.P_x, Self.P_v, Self.NC](
            ctx, num_x, num_v,
        )
        self.stage1 = PhaseState1D1V[Self.P_x, Self.P_v, Self.NC](
            ctx, num_x, num_v,
        )
        self.stage2 = PhaseState1D1V[Self.P_x, Self.P_v, Self.NC](
            ctx, num_x, num_v,
        )
        self.rhs = PhaseState1D1V[Self.P_x, Self.P_v, Self.NC](
            ctx, num_x, num_v,
        )

    def step[
        ComputeRhs: def(
            mut PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
            mut PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
        ) raises -> None,
        Adjust: def(
            mut PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
        dt: Real,
        compute_rhs_fn: ComputeRhs,
        adjust_fn: Adjust,
    ) raises:
        var size = u.buf_len
        comptime BLOCK = 256
        var grid = ceildiv(size, BLOCK)
        var layout = row_major(Int(size))

        adjust_fn(u)
        ctx.enqueue_copy(dst_buf=self.u_save.buf, src_buf=u.buf)

        compute_rhs_fn(self.u_save, self.rhs)
        comptime k_euler = _euler_step_kernel[
            type_of(layout), type_of(layout), type_of(layout),
        ]
        ctx.enqueue_function[k_euler](
            TileTensor(self.u_save.buf, layout),
            TileTensor(self.rhs.buf, layout),
            TileTensor(self.stage1.buf, layout),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        adjust_fn(self.stage1)

        comptime k1 = _ssprk3_combine_kernel[
            type_of(layout), type_of(layout), type_of(layout), type_of(layout),
        ]
        compute_rhs_fn(self.stage1, self.rhs)
        ctx.enqueue_function[k1](
            TileTensor(self.u_save.buf, layout),
            TileTensor(self.stage1.buf, layout),
            TileTensor(self.rhs.buf, layout),
            TileTensor(self.stage2.buf, layout),
            Real(0.75), Real(0.25), dt, Int64(size),
            grid_dim=grid, block_dim=BLOCK,
        )
        adjust_fn(self.stage2)

        compute_rhs_fn(self.stage2, self.rhs)
        ctx.enqueue_function[k1](
            TileTensor(self.u_save.buf, layout),
            TileTensor(self.stage2.buf, layout),
            TileTensor(self.rhs.buf, layout),
            TileTensor(u.buf, layout),
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0), dt, Int64(size),
            grid_dim=grid, block_dim=BLOCK,
        )
        adjust_fn(u)

    def run_until[
        ComputeRhs: def(
            mut PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
            mut PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
        ) raises -> None,
        Adjust: def(
            mut PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: PhaseState1D1V[Self.P_x, Self.P_v, Self.NC],
        t: Real,
        t_target: Real,
        dt: Real,
        compute_rhs_fn: ComputeRhs,
        adjust_fn: Adjust,
    ) raises -> Real:
        """Advance from time `t` to `t_target` in steps of `dt`, shortening
        the last step to land on `t_target` exactly. Returns the time
        reached, so a frame loop is `t = stepper.run_until(..., t, t_next,
        dt, ...)`."""
        var time = t
        # The tolerance stops round-off in `time` from adding a ~1e-16 step.
        while time < t_target - Real(1.0e-9) * dt:
            var dt_use = dt if (time + dt) < t_target else (t_target - time)
            self.step(ctx, u, dt_use, compute_rhs_fn, adjust_fn)
            time += dt_use
        return time
