# Joint SSPRK3 stepper for 1D2V Vlasov + 1D Maxwell.
#
# Structural twin of `ssprk3_vlasov_maxwell_1d1v.mojo` with the kinetic
# state widened to `PhaseState1D2V`. The Maxwell side stays a 1D config-
# space DG state — `E` and `B` are functions of `x` alone in this
# reduced model. Two-axis coupling (J_x → E_x, J_y → E_y) lives in the
# caller-supplied `compute_joint_rhs_fn` closure, not in this stepper.

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from std.math import ceildiv

from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state_1d import DGState1D


struct SSPRK3_VlasovMaxwell_1D2V[P_x: Int, P_v1: Int, P_v2: Int]:
    """SSPRK3 stepper for a coupled 1D2V Vlasov + 1D Maxwell system."""

    var qp_save: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC]
    var qp_stage1: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC]
    var qp_stage2: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC]
    var qp_rhs: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC]
    var qm_save: DGState1D[Self.P_x, MAXWELL_NC]
    var qm_stage1: DGState1D[Self.P_x, MAXWELL_NC]
    var qm_stage2: DGState1D[Self.P_x, MAXWELL_NC]
    var qm_rhs: DGState1D[Self.P_x, MAXWELL_NC]

    def __init__(
        out self,
        var ctx: DeviceContext,
        num_x: Int, num_v1: Int, num_v2: Int,
    ) raises:
        self.qp_save = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ](ctx, num_x, num_v1, num_v2)
        self.qp_stage1 = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ](ctx, num_x, num_v1, num_v2)
        self.qp_stage2 = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ](ctx, num_x, num_v1, num_v2)
        self.qp_rhs = PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ](ctx, num_x, num_v1, num_v2)
        self.qm_save = DGState1D[Self.P_x, MAXWELL_NC](ctx, num_x)
        self.qm_stage1 = DGState1D[Self.P_x, MAXWELL_NC](ctx, num_x)
        self.qm_stage2 = DGState1D[Self.P_x, MAXWELL_NC](ctx, num_x)
        self.qm_rhs = DGState1D[Self.P_x, MAXWELL_NC](ctx, num_x)

    def step[
        ComputeJointRhs: def(
            mut PhaseState1D2V[
                Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
            ],
            mut DGState1D[Self.P_x, MAXWELL_NC],
            mut PhaseState1D2V[
                Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
            ],
            mut DGState1D[Self.P_x, MAXWELL_NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_phase: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut q_maxwell: DGState1D[Self.P_x, MAXWELL_NC],
        dt: Real,
        compute_joint_rhs_fn: ComputeJointRhs,
    ) raises:
        var p_size = q_phase.buf_len
        var m_size = q_maxwell.buf_len
        comptime BLOCK = 256
        var p_grid = ceildiv(p_size, BLOCK)
        var m_grid = ceildiv(m_size, BLOCK)
        var pl = row_major(Int(p_size))
        var ml = row_major(Int(m_size))

        ctx.enqueue_copy(dst_buf=self.qp_save.buf, src_buf=q_phase.buf)
        ctx.enqueue_copy(dst_buf=self.qm_save.buf, src_buf=q_maxwell.buf)

        comptime k_euler_p = _euler_step_kernel[
            type_of(pl), type_of(pl), type_of(pl),
        ]
        comptime k_euler_m = _euler_step_kernel[
            type_of(ml), type_of(ml), type_of(ml),
        ]
        comptime k_combine_p = _ssprk3_combine_kernel[
            type_of(pl), type_of(pl), type_of(pl), type_of(pl),
        ]
        comptime k_combine_m = _ssprk3_combine_kernel[
            type_of(ml), type_of(ml), type_of(ml), type_of(ml),
        ]

        # Stage 1: u^(1) = u^n + dt L(u^n)
        compute_joint_rhs_fn(
            self.qp_save, self.qm_save,
            self.qp_rhs, self.qm_rhs,
        )
        ctx.enqueue_function[k_euler_p](
            TileTensor(self.qp_save.buf, pl),
            TileTensor(self.qp_rhs.buf, pl),
            TileTensor(self.qp_stage1.buf, pl),
            dt, Int64(p_size), grid_dim=p_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_euler_m](
            TileTensor(self.qm_save.buf, ml),
            TileTensor(self.qm_rhs.buf, ml),
            TileTensor(self.qm_stage1.buf, ml),
            dt, Int64(m_size), grid_dim=m_grid, block_dim=BLOCK,
        )

        # Stage 2: u^(2) = 3/4 u^n + 1/4 (u^(1) + dt L(u^(1)))
        compute_joint_rhs_fn(
            self.qp_stage1, self.qm_stage1,
            self.qp_rhs, self.qm_rhs,
        )
        ctx.enqueue_function[k_combine_p](
            TileTensor(self.qp_save.buf, pl),
            TileTensor(self.qp_stage1.buf, pl),
            TileTensor(self.qp_rhs.buf, pl),
            TileTensor(self.qp_stage2.buf, pl),
            Real(0.75), Real(0.25), dt, Int64(p_size),
            grid_dim=p_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_combine_m](
            TileTensor(self.qm_save.buf, ml),
            TileTensor(self.qm_stage1.buf, ml),
            TileTensor(self.qm_rhs.buf, ml),
            TileTensor(self.qm_stage2.buf, ml),
            Real(0.75), Real(0.25), dt, Int64(m_size),
            grid_dim=m_grid, block_dim=BLOCK,
        )

        # Stage 3: u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt L(u^(2)))
        compute_joint_rhs_fn(
            self.qp_stage2, self.qm_stage2,
            self.qp_rhs, self.qm_rhs,
        )
        ctx.enqueue_function[k_combine_p](
            TileTensor(self.qp_save.buf, pl),
            TileTensor(self.qp_stage2.buf, pl),
            TileTensor(self.qp_rhs.buf, pl),
            TileTensor(q_phase.buf, pl),
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0), dt, Int64(p_size),
            grid_dim=p_grid, block_dim=BLOCK,
        )
        ctx.enqueue_function[k_combine_m](
            TileTensor(self.qm_save.buf, ml),
            TileTensor(self.qm_stage2.buf, ml),
            TileTensor(self.qm_rhs.buf, ml),
            TileTensor(q_maxwell.buf, ml),
            Real(1.0) / Real(3.0), Real(2.0) / Real(3.0), dt, Int64(m_size),
            grid_dim=m_grid, block_dim=BLOCK,
        )
