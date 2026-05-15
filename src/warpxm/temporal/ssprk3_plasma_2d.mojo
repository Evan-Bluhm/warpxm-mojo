# Joint SSPRK3 stepper for a two-species plasma: two 5-moment Euler fluids
# (e.g. electrons + ions) coupled to a shared Maxwell field.
#
# Each species has its own charge-to-mass ratio α_s. Per RK stage:
#   (1) Compute spatial RHS of fluid_1, fluid_2, Maxwell (separately).
#   (2) Apply the Lorentz source to each fluid using the current Maxwell state.
#   (3) Accumulate both species' currents onto Maxwell's E-RHS (one
#       `apply_current_source_2d` call per species).
#   (4) Combine each field with the standard SSPRK3 weights.
#
# Setting `α_1 = -1, α_2 = +1` with equal masses and equal background
# density reproduces the simplest "symmetric pair-plasma" oscillation at
# `ω = √(α_1²·n_1 + α_2²·n_2) = √2 · ω_pe`. For a realistic electron+ion
# plasma (proton-to-electron mass ratio ~ 1836), set the ion fluid's α to
# ~ 1/1836 — the ion contribution to the plasma frequency is then
# negligible, and the system reduces to the single-species Langmuir
# oscillation around an effectively static ion background.

from std.gpu.host import DeviceContext
from layout import TileTensor, Idx, row_major
from std.math import ceildiv

from warpxm.apps.euler_2d import EULER_NC
from warpxm.apps.maxwell_2d import MAXWELL_NC
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.spatial.lorentz_source_2d import apply_lorentz_source_2d
from warpxm.spatial.maxwell_current_source_2d import apply_current_source_2d
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.state_2d import DGState2D


struct SSPRK3_Plasma_2D[P: Int]:
    """SSPRK3 stepper for two fluid species + Maxwell with full plasma coupling."""

    # Species-1 scratch
    var u1_save: DGState2D[Self.P, EULER_NC]
    var s1_stage1: DGState2D[Self.P, EULER_NC]
    var s1_stage2: DGState2D[Self.P, EULER_NC]
    var s1_rhs: DGState2D[Self.P, EULER_NC]
    # Species-2 scratch
    var u2_save: DGState2D[Self.P, EULER_NC]
    var s2_stage1: DGState2D[Self.P, EULER_NC]
    var s2_stage2: DGState2D[Self.P, EULER_NC]
    var s2_rhs: DGState2D[Self.P, EULER_NC]
    # Maxwell scratch
    var um_save: DGState2D[Self.P, MAXWELL_NC]
    var m_stage1: DGState2D[Self.P, MAXWELL_NC]
    var m_stage2: DGState2D[Self.P, MAXWELL_NC]
    var m_rhs: DGState2D[Self.P, MAXWELL_NC]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u1_save = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.s1_stage1 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.s1_stage2 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.s1_rhs = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.u2_save = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.s2_stage1 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.s2_stage2 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.s2_rhs = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.um_save = DGState2D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_stage1 = DGState2D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_stage2 = DGState2D[Self.P, MAXWELL_NC](ctx, num_elements)
        self.m_rhs = DGState2D[Self.P, MAXWELL_NC](ctx, num_elements)

    def step(
        mut self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        fluid_spatial: DGEulerSolver2D[Self.P],
        maxwell_spatial: DGMaxwellSolver2D[Self.P],
        mut q_s1: DGState2D[Self.P, EULER_NC],
        mut q_s2: DGState2D[Self.P, EULER_NC],
        mut q_maxwell: DGState2D[Self.P, MAXWELL_NC],
        alpha_s1: Real,
        alpha_s2: Real,
        eps0: Real,
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

        # --- Stage 1 -----------------------------------------------------------
        fluid_spatial.compute_rhs(ctx, mesh, self.u1_save, self.s1_rhs)
        fluid_spatial.compute_rhs(ctx, mesh, self.u2_save, self.s2_rhs)
        maxwell_spatial.compute_rhs(ctx, mesh, self.um_save, self.m_rhs)
        apply_lorentz_source_2d[Self.P](
            ctx, self.u1_save, self.um_save, self.s1_rhs, alpha_s1
        )
        apply_lorentz_source_2d[Self.P](
            ctx, self.u2_save, self.um_save, self.s2_rhs, alpha_s2
        )
        apply_current_source_2d[Self.P](
            ctx, self.u1_save, self.m_rhs, alpha_s1, eps0
        )
        apply_current_source_2d[Self.P](
            ctx, self.u2_save, self.m_rhs, alpha_s2, eps0
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

        # --- Stage 2 -----------------------------------------------------------
        fluid_spatial.compute_rhs(ctx, mesh, self.s1_stage1, self.s1_rhs)
        fluid_spatial.compute_rhs(ctx, mesh, self.s2_stage1, self.s2_rhs)
        maxwell_spatial.compute_rhs(ctx, mesh, self.m_stage1, self.m_rhs)
        apply_lorentz_source_2d[Self.P](
            ctx, self.s1_stage1, self.m_stage1, self.s1_rhs, alpha_s1
        )
        apply_lorentz_source_2d[Self.P](
            ctx, self.s2_stage1, self.m_stage1, self.s2_rhs, alpha_s2
        )
        apply_current_source_2d[Self.P](
            ctx, self.s1_stage1, self.m_rhs, alpha_s1, eps0
        )
        apply_current_source_2d[Self.P](
            ctx, self.s2_stage1, self.m_rhs, alpha_s2, eps0
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

        # --- Stage 3 -----------------------------------------------------------
        fluid_spatial.compute_rhs(ctx, mesh, self.s1_stage2, self.s1_rhs)
        fluid_spatial.compute_rhs(ctx, mesh, self.s2_stage2, self.s2_rhs)
        maxwell_spatial.compute_rhs(ctx, mesh, self.m_stage2, self.m_rhs)
        apply_lorentz_source_2d[Self.P](
            ctx, self.s1_stage2, self.m_stage2, self.s1_rhs, alpha_s1
        )
        apply_lorentz_source_2d[Self.P](
            ctx, self.s2_stage2, self.m_stage2, self.s2_rhs, alpha_s2
        )
        apply_current_source_2d[Self.P](
            ctx, self.s1_stage2, self.m_rhs, alpha_s1, eps0
        )
        apply_current_source_2d[Self.P](
            ctx, self.s2_stage2, self.m_rhs, alpha_s2, eps0
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
