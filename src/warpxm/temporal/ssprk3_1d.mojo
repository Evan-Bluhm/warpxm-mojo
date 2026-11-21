# Generic SSPRK3 stepper for 1D DG fields. Pure compositional analogue of
# `ssprk3_2d.mojo`: caller supplies `compute_rhs` and `adjust` closures
# that capture whichever spatial solver, mesh, and adjusters the driver
# uses.

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.math import ceildiv

from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.state_1d import DGState1D


struct SSPRK3_1D[P: Int, NC: Int]:
    """Generic SSPRK3 scratch + stepper for any 1D NC-component DG problem."""

    var u_save: DGState1D[Self.P, Self.NC]
    var stage1: DGState1D[Self.P, Self.NC]
    var stage2: DGState1D[Self.P, Self.NC]
    var rhs: DGState1D[Self.P, Self.NC]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u_save = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.stage1 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.stage2 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.rhs = DGState1D[Self.P, Self.NC](ctx, num_elements)

    def step[
        ComputeRhs: def(
            mut DGState1D[Self.P, Self.NC],
            mut DGState1D[Self.P, Self.NC],
        ) raises -> None,
        Adjust: def(mut DGState1D[Self.P, Self.NC]) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState1D[Self.P, Self.NC],
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
