# Joint SSPRK3 stepper for 1D1V Vlasov + 1D Maxwell.
#
# Mirrors `ssprk3_plasma_1d.mojo`: a single time-step that advances both
# the phase-space distribution `f(x, v)` and the configuration-space
# Maxwell aux state through three SSPRK3 stages. The caller supplies a
# `compute_joint_rhs_fn` closure that — given a (q_phase, q_maxwell)
# input pair — fills (rhs_phase, rhs_maxwell). Inside that closure the
# driver is responsible for:
#
#   1. Vlasov RHS (`DGVlasovSolver1D1V.compute_rhs(..., q_maxwell)`)
#   2. Maxwell RHS (`DGMaxwellSolver[1, P].compute_rhs(...)`)
#   3. Compute species current `J = ∫ v f dv` via
#      `VlasovMoments1D1V.first_moment(...)`.
#   4. Add current source: `apply_vlasov_current_source_1d(j,
#      rhs_maxwell, α, ε₀)`.
#
# This keeps the stepper itself ignorant of coupling — same pattern as
# the upstream WARPXM kernel-launcher registry — and lets new sources
# (e.g. BGK collisions, ponderomotive force) be composed by the
# driver without changing the stepper.

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from std.math import ceildiv

from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state import DGState


struct SSPRK3_VlasovMaxwell_1D1V[P_x: Int, P_v: Int]:
    """SSPRK3 stepper for a coupled 1D1V Vlasov + 1D Maxwell system."""

    var qp_save: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC]
    var qp_stage1: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC]
    var qp_stage2: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC]
    var qp_rhs: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC]
    var qm_save: DGState[1, Self.P_x, MAXWELL_NC]
    var qm_stage1: DGState[1, Self.P_x, MAXWELL_NC]
    var qm_stage2: DGState[1, Self.P_x, MAXWELL_NC]
    var qm_rhs: DGState[1, Self.P_x, MAXWELL_NC]

    def __init__(
        out self, var ctx: DeviceContext, num_x: Int, num_v: Int,
    ) raises:
        self.qp_save = PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC](
            ctx, num_x, num_v,
        )
        self.qp_stage1 = PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC](
            ctx, num_x, num_v,
        )
        self.qp_stage2 = PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC](
            ctx, num_x, num_v,
        )
        self.qp_rhs = PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC](
            ctx, num_x, num_v,
        )
        self.qm_save = DGState[1, Self.P_x, MAXWELL_NC](ctx, num_x)
        self.qm_stage1 = DGState[1, Self.P_x, MAXWELL_NC](ctx, num_x)
        self.qm_stage2 = DGState[1, Self.P_x, MAXWELL_NC](ctx, num_x)
        self.qm_rhs = DGState[1, Self.P_x, MAXWELL_NC](ctx, num_x)

    def step[
        ComputeJointRhs: def(
            mut PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
            mut DGState[1, Self.P_x, MAXWELL_NC],
            mut PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
            mut DGState[1, Self.P_x, MAXWELL_NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_phase: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        mut q_maxwell: DGState[1, Self.P_x, MAXWELL_NC],
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

    def run_until[
        ComputeJointRhs: def(
            mut PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
            mut DGState[1, Self.P_x, MAXWELL_NC],
            mut PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
            mut DGState[1, Self.P_x, MAXWELL_NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_phase: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        mut q_maxwell: DGState[1, Self.P_x, MAXWELL_NC],
        t: Real,
        t_target: Real,
        dt: Real,
        compute_joint_rhs_fn: ComputeJointRhs,
    ) raises -> Real:
        """Advance from time `t` to `t_target` in steps of `dt`, shortening
        the last step to land on `t_target` exactly. Returns the time
        reached, so a frame loop is `t = stepper.run_until(..., t, t_next,
        dt, ...)`."""
        var time = t
        # The tolerance stops round-off in `time` from adding a ~1e-16 step.
        while time < t_target - Real(1.0e-9) * dt:
            var dt_use = dt if (time + dt) < t_target else (t_target - time)
            self.step(ctx, q_phase, q_maxwell, dt_use, compute_joint_rhs_fn)
            time += dt_use
        return time
