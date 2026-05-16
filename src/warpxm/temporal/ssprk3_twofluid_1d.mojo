# Joint SSPRK3 stepper for a 1D 5-moment Euler fluid coupled to Maxwell.
# 1D analogue of `ssprk3_twofluid_2d.mojo`.

from std.gpu.host import DeviceContext
from layout import TileTensor, Idx, row_major
from std.math import ceildiv

from warpxm.apps.euler import EULER_NC
from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.state_1d import DGState1D


struct SSPRK3_TwoFluid_1D[P: Int]:
    """SSPRK3 stepper for a 1D fluid + Maxwell pair."""

    var uf_save: DGState1D[Self.P, EULER_NC]
    var f_stage1: DGState1D[Self.P, EULER_NC]
    var f_stage2: DGState1D[Self.P, EULER_NC]
    var f_rhs: DGState1D[Self.P, EULER_NC]
    var um_save: DGState1D[Self.P, MAXWELL_NC]
    var m_stage1: DGState1D[Self.P, MAXWELL_NC]
    var m_stage2: DGState1D[Self.P, MAXWELL_NC]
    var m_rhs: DGState1D[Self.P, MAXWELL_NC]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.uf_save = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.f_stage1 = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.f_stage2 = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.f_rhs = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.um_save = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_stage1 = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_stage2 = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_rhs = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)

    def step[
        compute_joint_rhs_fn: def(
            mut DGState1D[Self.P, EULER_NC],
            mut DGState1D[Self.P, MAXWELL_NC],
            mut DGState1D[Self.P, EULER_NC],
            mut DGState1D[Self.P, MAXWELL_NC],
        ) raises capturing [_] -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_fluid: DGState1D[Self.P, EULER_NC],
        mut q_maxwell: DGState1D[Self.P, MAXWELL_NC],
        dt: Real,
    ) raises:
        var f_size = q_fluid.buf_len
        var m_size = q_maxwell.buf_len
        comptime BLOCK = 256
        var f_grid = ceildiv(f_size, BLOCK)
        var m_grid = ceildiv(m_size, BLOCK)
        var fl = row_major(Idx(f_size))
        var ml = row_major(Idx(m_size))

        ctx.enqueue_copy(dst_buf=self.uf_save.buf, src_buf=q_fluid.buf)
        ctx.enqueue_copy(dst_buf=self.um_save.buf, src_buf=q_maxwell.buf)

        comptime k_euler_f = _euler_step_kernel[
            type_of(fl), type_of(fl), type_of(fl),
        ]
        comptime k_euler_m = _euler_step_kernel[
            type_of(ml), type_of(ml), type_of(ml),
        ]
        comptime k_combine_f = _ssprk3_combine_kernel[
            type_of(fl), type_of(fl), type_of(fl), type_of(fl),
        ]
        comptime k_combine_m = _ssprk3_combine_kernel[
            type_of(ml), type_of(ml), type_of(ml), type_of(ml),
        ]

        compute_joint_rhs_fn(
            self.uf_save, self.um_save, self.f_rhs, self.m_rhs,
        )
        ctx.enqueue_function[k_euler_f](
            TileTensor(self.uf_save.buf, fl),
            TileTensor(self.f_rhs.buf, fl),
            TileTensor(self.f_stage1.buf, fl),
            dt, f_size, grid_dim=f_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_euler_m](
            TileTensor(self.um_save.buf, ml),
            TileTensor(self.m_rhs.buf, ml),
            TileTensor(self.m_stage1.buf, ml),
            dt, m_size, grid_dim=m_grid, block_dim=BLOCK,
        )

        compute_joint_rhs_fn(
            self.f_stage1, self.m_stage1, self.f_rhs, self.m_rhs,
        )
        ctx.enqueue_function[k_combine_f](
            TileTensor(self.uf_save.buf, fl),
            TileTensor(self.f_stage1.buf, fl),
            TileTensor(self.f_rhs.buf, fl),
            TileTensor(self.f_stage2.buf, fl),
            Real(0.75), Real(0.25), dt, f_size,
            grid_dim=f_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_combine_m](
            TileTensor(self.um_save.buf, ml),
            TileTensor(self.m_stage1.buf, ml),
            TileTensor(self.m_rhs.buf, ml),
            TileTensor(self.m_stage2.buf, ml),
            Real(0.75), Real(0.25), dt, m_size,
            grid_dim=m_grid, block_dim=BLOCK,
        )

        compute_joint_rhs_fn(
            self.f_stage2, self.m_stage2, self.f_rhs, self.m_rhs,
        )
        ctx.enqueue_function[k_combine_f](
            TileTensor(self.uf_save.buf, fl),
            TileTensor(self.f_stage2.buf, fl),
            TileTensor(self.f_rhs.buf, fl),
            TileTensor(q_fluid.buf, fl),
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0), dt, f_size,
            grid_dim=f_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_combine_m](
            TileTensor(self.um_save.buf, ml),
            TileTensor(self.m_stage2.buf, ml),
            TileTensor(self.m_rhs.buf, ml),
            TileTensor(q_maxwell.buf, ml),
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0), dt, m_size,
            grid_dim=m_grid, block_dim=BLOCK,
        )
