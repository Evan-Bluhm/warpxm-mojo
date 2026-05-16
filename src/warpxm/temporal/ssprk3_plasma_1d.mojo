# Joint SSPRK3 stepper for two coupled fluid species + Maxwell, 1D.
# 1D analogue of `ssprk3_plasma_2d.mojo`. Same closure pattern.

from std.gpu.host import DeviceContext
from layout import TileTensor, Idx, row_major
from std.math import ceildiv

from warpxm.apps.euler import EULER_NC
from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.state_1d import DGState1D


struct SSPRK3_Plasma_1D[P: Int]:
    """SSPRK3 stepper for two 1D fluid species + Maxwell."""

    var u1_save: DGState1D[Self.P, EULER_NC]
    var s1_stage1: DGState1D[Self.P, EULER_NC]
    var s1_stage2: DGState1D[Self.P, EULER_NC]
    var s1_rhs: DGState1D[Self.P, EULER_NC]
    var u2_save: DGState1D[Self.P, EULER_NC]
    var s2_stage1: DGState1D[Self.P, EULER_NC]
    var s2_stage2: DGState1D[Self.P, EULER_NC]
    var s2_rhs: DGState1D[Self.P, EULER_NC]
    var um_save: DGState1D[Self.P, MAXWELL_NC]
    var m_stage1: DGState1D[Self.P, MAXWELL_NC]
    var m_stage2: DGState1D[Self.P, MAXWELL_NC]
    var m_rhs: DGState1D[Self.P, MAXWELL_NC]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u1_save = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.s1_stage1 = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.s1_stage2 = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.s1_rhs = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.u2_save = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.s2_stage1 = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.s2_stage2 = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.s2_rhs = DGState1D[Self.P, EULER_NC](ctx, num_elements)
        self.um_save = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_stage1 = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_stage2 = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_rhs = DGState1D[Self.P, MAXWELL_NC](ctx, num_elements)

    def step[
        compute_joint_rhs_fn: def(
            mut DGState1D[Self.P, EULER_NC],
            mut DGState1D[Self.P, EULER_NC],
            mut DGState1D[Self.P, MAXWELL_NC],
            mut DGState1D[Self.P, EULER_NC],
            mut DGState1D[Self.P, EULER_NC],
            mut DGState1D[Self.P, MAXWELL_NC],
        ) raises capturing [_] -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_s1: DGState1D[Self.P, EULER_NC],
        mut q_s2: DGState1D[Self.P, EULER_NC],
        mut q_maxwell: DGState1D[Self.P, MAXWELL_NC],
        dt: Real,
    ) raises:
        var f_size = q_s1.buf_len
        var m_size = q_maxwell.buf_len
        comptime BLOCK = 256
        var f_grid = ceildiv(f_size, BLOCK)
        var m_grid = ceildiv(m_size, BLOCK)
        var fl = row_major(Idx(f_size))
        var ml = row_major(Idx(m_size))

        ctx.enqueue_copy(dst_buf=self.u1_save.buf, src_buf=q_s1.buf)
        ctx.enqueue_copy(dst_buf=self.u2_save.buf, src_buf=q_s2.buf)
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
            self.u1_save, self.u2_save, self.um_save,
            self.s1_rhs, self.s2_rhs, self.m_rhs,
        )
        ctx.enqueue_function[k_euler_f](
            TileTensor(self.u1_save.buf, fl),
            TileTensor(self.s1_rhs.buf, fl),
            TileTensor(self.s1_stage1.buf, fl),
            dt, f_size, grid_dim=f_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_euler_f](
            TileTensor(self.u2_save.buf, fl),
            TileTensor(self.s2_rhs.buf, fl),
            TileTensor(self.s2_stage1.buf, fl),
            dt, f_size, grid_dim=f_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_euler_m](
            TileTensor(self.um_save.buf, ml),
            TileTensor(self.m_rhs.buf, ml),
            TileTensor(self.m_stage1.buf, ml),
            dt, m_size, grid_dim=m_grid, block_dim=BLOCK,
        )

        compute_joint_rhs_fn(
            self.s1_stage1, self.s2_stage1, self.m_stage1,
            self.s1_rhs, self.s2_rhs, self.m_rhs,
        )
        ctx.enqueue_function[k_combine_f](
            TileTensor(self.u1_save.buf, fl),
            TileTensor(self.s1_stage1.buf, fl),
            TileTensor(self.s1_rhs.buf, fl),
            TileTensor(self.s1_stage2.buf, fl),
            Real(0.75), Real(0.25), dt, f_size,
            grid_dim=f_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_combine_f](
            TileTensor(self.u2_save.buf, fl),
            TileTensor(self.s2_stage1.buf, fl),
            TileTensor(self.s2_rhs.buf, fl),
            TileTensor(self.s2_stage2.buf, fl),
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
            self.s1_stage2, self.s2_stage2, self.m_stage2,
            self.s1_rhs, self.s2_rhs, self.m_rhs,
        )
        ctx.enqueue_function[k_combine_f](
            TileTensor(self.u1_save.buf, fl),
            TileTensor(self.s1_stage2.buf, fl),
            TileTensor(self.s1_rhs.buf, fl),
            TileTensor(q_s1.buf, fl),
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0), dt, f_size,
            grid_dim=f_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_combine_f](
            TileTensor(self.u2_save.buf, fl),
            TileTensor(self.s2_stage2.buf, fl),
            TileTensor(self.s2_rhs.buf, fl),
            TileTensor(q_s2.buf, fl),
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
