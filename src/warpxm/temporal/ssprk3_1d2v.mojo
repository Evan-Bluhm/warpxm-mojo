# Generic SSPRK3 stepper for 1D2V phase-space DG fields.
#
# Mirrors `temporal/ssprk3_1d1v.mojo` but parameterised by separate
# physical-space and two-velocity-axis orders `(P_x, P_v1, P_v2)` and
# operating on `PhaseState1D2V` buffers. The flat-buffer combine
# kernels (`_euler_step_kernel`, `_ssprk3_combine_kernel` in
# `ssprk3.mojo`) are unchanged — they don't care about dimensionality.

from std.gpu.host import DeviceContext
from layout import TileTensor, Idx, row_major
from std.math import ceildiv

from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.phase_state_1d2v import PhaseState1D2V


@always_inline
def no_op_adjust_1d2v[P_x: Int, P_v1: Int, P_v2: Int, NC: Int](
    mut q: PhaseState1D2V[P_x, P_v1, P_v2, NC],
) raises:
    """Identity adjuster — pass this when no adjuster step is wanted.
    The Vlasov-Maxwell port does not currently ship a phase-space
    limiter, so this is the only adjuster every 1D2V driver needs
    for now."""
    pass


struct SSPRK3_1D2V[P_x: Int, P_v1: Int, P_v2: Int, NC: Int]:
    """Three-stage Shu-Osher SSPRK3 stepper on a 1D2V phase-space field."""

    var u_save: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, Self.NC]
    var stage1: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, Self.NC]
    var stage2: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, Self.NC]
    var rhs: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, Self.NC]

    def __init__(
        out self,
        var ctx: DeviceContext,
        num_x: Int, num_v1: Int, num_v2: Int,
    ) raises:
        self.u_save = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, Self.NC,
        ](ctx, num_x, num_v1, num_v2)
        self.stage1 = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, Self.NC,
        ](ctx, num_x, num_v1, num_v2)
        self.stage2 = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, Self.NC,
        ](ctx, num_x, num_v1, num_v2)
        self.rhs = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, Self.NC,
        ](ctx, num_x, num_v1, num_v2)

    def step[
        compute_rhs_fn: def(
            mut PhaseState1D2V[
                Self.P_x, Self.P_v1, Self.P_v2, Self.NC,
            ],
            mut PhaseState1D2V[
                Self.P_x, Self.P_v1, Self.P_v2, Self.NC,
            ],
        ) raises capturing [_] -> None,
        adjust_fn: def(
            mut PhaseState1D2V[
                Self.P_x, Self.P_v1, Self.P_v2, Self.NC,
            ],
        ) raises capturing [_] -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, Self.NC],
        dt: Real,
    ) raises:
        var size = u.buf_len
        comptime BLOCK = 256
        var grid = ceildiv(size, BLOCK)
        var layout = row_major(Idx(size))

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
            dt, size, grid_dim=grid, block_dim=BLOCK,
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
            Real(0.75), Real(0.25), dt, size,
            grid_dim=grid, block_dim=BLOCK,
        )
        adjust_fn(self.stage2)

        compute_rhs_fn(self.stage2, self.rhs)
        ctx.enqueue_function[k1](
            TileTensor(self.u_save.buf, layout),
            TileTensor(self.stage2.buf, layout),
            TileTensor(self.rhs.buf, layout),
            TileTensor(u.buf, layout),
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0), dt, size,
            grid_dim=grid, block_dim=BLOCK,
        )
        adjust_fn(u)
