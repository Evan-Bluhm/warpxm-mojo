# Joint SSPRK3 stepper for Vlasov + Maxwell, generic over the dimensions.
#
# Advances a kinetic `PhaseState[XDIM, VDIM, P_x, P_v, VLASOV_NC]` and the
# configuration-space Maxwell `DGState[XDIM, P_x, MAXWELL_NC]` through the
# same three Shu-Osher stages. The coupling (Lorentz force on `f`, current
# into Ampère's law) lives in the caller's `compute_joint_rhs_fn(q_phase,
# q_maxwell, rhs_phase, rhs_maxwell)` closure, not in the stepper.
#
# Like `SSPRK3Phase` it keeps two scratch copies of each state (a stage
# value and the right-hand side) and overwrites the stages in place.

from max.gpu.host import DeviceContext

from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.core.types import Real
from warpxm.temporal.pacing import CflPacer
from warpxm.temporal.ssprk3_phase import (
    ssprk3_euler_stage, ssprk3_inplace_stage,
)
from warpxm.variable.phase_state import PhaseState
from warpxm.variable.state import DGState


struct SSPRK3VlasovMaxwell[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int]:
    """SSPRK3 stepper for a coupled Vlasov + Maxwell system."""

    comptime Phase = PhaseState[
        Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, VLASOV_NC,
    ]
    comptime Fields = DGState[Self.XDIM, Self.P_x, MAXWELL_NC]

    var qp_stage: Self.Phase
    var qp_rhs: Self.Phase
    var qm_stage: Self.Fields
    var qm_rhs: Self.Fields

    def __init__(
        out self,
        var ctx: DeviceContext,
        num_phase_elements: Int,
        num_x_elements: Int,
        num_phase_ghost: Int = 0,
        num_x_ghost: Int = 0,
    ) raises:
        """Pass `mesh.num_phase_elements`, `mesh.num_x_elements` and, for
        a run on several ranks, the ghost counts of the phase mesh
        (`mesh.num_ghost`) and of the field mesh (`x_mesh.num_ghost`)."""
        self.qp_stage = Self.Phase(ctx, num_phase_elements, num_phase_ghost)
        self.qp_rhs = Self.Phase(ctx, num_phase_elements, num_phase_ghost)
        self.qm_stage = Self.Fields(ctx, num_x_elements, num_x_ghost)
        self.qm_rhs = Self.Fields(ctx, num_x_elements, num_x_ghost)

    def step[
        ComputeJointRhs: def(
            mut Self.Phase, mut Self.Fields, mut Self.Phase, mut Self.Fields,
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_phase: Self.Phase,
        mut q_maxwell: Self.Fields,
        dt: Real,
        compute_joint_rhs_fn: ComputeJointRhs,
    ) raises:
        var p_size = q_phase.buf_len
        var m_size = q_maxwell.buf_len
        comptime THIRD: Real = 1.0 / 3.0
        comptime TWO_THIRDS: Real = 2.0 / 3.0

        # Stage 1: w = u^n + dt L(u^n)
        compute_joint_rhs_fn(q_phase, q_maxwell, self.qp_rhs, self.qm_rhs)
        ssprk3_euler_stage(
            ctx, q_phase.buf, self.qp_rhs.buf, self.qp_stage.buf, dt, p_size,
        )
        ssprk3_euler_stage(
            ctx, q_maxwell.buf, self.qm_rhs.buf, self.qm_stage.buf,
            dt, m_size,
        )

        # Stage 2: w = 1/4 (w + dt L(w)) + 3/4 u^n
        compute_joint_rhs_fn(
            self.qp_stage, self.qm_stage, self.qp_rhs, self.qm_rhs,
        )
        ssprk3_inplace_stage(
            ctx, self.qp_stage.buf, q_phase.buf, self.qp_rhs.buf,
            Real(0.25), Real(0.75), Real(0.25) * dt, p_size,
        )
        ssprk3_inplace_stage(
            ctx, self.qm_stage.buf, q_maxwell.buf, self.qm_rhs.buf,
            Real(0.25), Real(0.75), Real(0.25) * dt, m_size,
        )

        # Stage 3: u^{n+1} = 1/3 u^n + 2/3 (w + dt L(w))
        compute_joint_rhs_fn(
            self.qp_stage, self.qm_stage, self.qp_rhs, self.qm_rhs,
        )
        ssprk3_inplace_stage(
            ctx, q_phase.buf, self.qp_stage.buf, self.qp_rhs.buf,
            THIRD, TWO_THIRDS, TWO_THIRDS * dt, p_size,
        )
        ssprk3_inplace_stage(
            ctx, q_maxwell.buf, self.qm_stage.buf, self.qm_rhs.buf,
            THIRD, TWO_THIRDS, TWO_THIRDS * dt, m_size,
        )

    def run_until[
        ComputeJointRhs: def(
            mut Self.Phase, mut Self.Fields, mut Self.Phase, mut Self.Fields,
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_phase: Self.Phase,
        mut q_maxwell: Self.Fields,
        t: Real,
        t_target: Real,
        dt: Real,
        compute_joint_rhs_fn: ComputeJointRhs,
    ) raises -> Real:
        """Advance from time `t` to `t_target` in steps of `dt`, shortening
        the last step to land on `t_target` exactly. Returns the time
        reached."""
        var time = t
        # The tolerance stops round-off in `time` from adding a ~1e-16 step.
        while time < t_target - Real(1.0e-9) * dt:
            var dt_use = dt if (time + dt) < t_target else (t_target - time)
            self.step(ctx, q_phase, q_maxwell, dt_use, compute_joint_rhs_fn)
            time += dt_use
        return time

    def run_until_cfl[
        ComputeJointRhs: def(
            mut Self.Phase, mut Self.Fields, mut Self.Phase, mut Self.Fields,
        ) raises -> None,
        SuggestDt: def(mut Self.Phase, mut Self.Fields) raises -> Real,
    ](
        mut self,
        var ctx: DeviceContext,
        mut q_phase: Self.Phase,
        mut q_maxwell: Self.Fields,
        t: Real,
        t_target: Real,
        compute_joint_rhs_fn: ComputeJointRhs,
        suggest_dt_fn: SuggestDt,
        dt_interval: Int = 1,
        dt_min: Real = 0.0,
    ) raises -> Real:
        """Advance from `t` to `t_target` with the step chosen by
        `suggest_dt_fn(q_phase, q_maxwell)` — typically the minimum of the
        Vlasov and Maxwell solvers' `suggest_dt` — refreshed every
        `dt_interval` steps. Raises if the suggestion is not above
        `dt_min`. Returns the time reached."""
        var time = t
        var pacer = CflPacer(dt_interval, dt_min)
        while time < t_target:
            if pacer.due():
                pacer.refresh(suggest_dt_fn(q_phase, q_maxwell), time)
            var remaining = t_target - time
            var dt_use = pacer.next(remaining)
            self.step(ctx, q_phase, q_maxwell, dt_use, compute_joint_rhs_fn)
            if dt_use == remaining:
                return t_target
            time += dt_use
        return time
