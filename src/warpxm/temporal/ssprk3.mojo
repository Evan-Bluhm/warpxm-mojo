# Generic SSPRK3 (three-stage Shu-Osher strong-stability-preserving
# Runge-Kutta) stepper for DG fields of any dimension, driven by two closures that
# the caller passes as trailing arguments:
#
#   * `compute_rhs_fn(q_in, q_out)` — evaluates the spatial DG residual
#     for the supplied input state and writes it to `q_out`. Captures
#     whichever spatial solver / mesh / device context the driver is
#     using.
#   * `adjust_fn(q)` — runs after each stage value is written; calls
#     limiter / positivity / nothing as the problem demands.
#
# One stepper serves every physics: adjuster composition is done at the
# call site rather than baked into a per-physics stepper.
#
# Pattern (driver code):
#
#     var spatial = DGEulerSolver[2, P](app)
#     var limiter = MoeRossmanith[2, P, EULER_NC](ctx, num_elements)
#     var stepper = SSPRK3[2, P, EULER_NC](ctx, num_elements)
#
#     def compute_rhs(
#         mut q_in: DGState[2, P, EULER_NC],
#         mut q_out: DGState[2, P, EULER_NC],
#     ) raises {imm}:
#         spatial.compute_rhs(ctx, mesh, q_in, q_out)
#
#     def adjust(mut q: DGState[2, P, EULER_NC]) raises {mut limiter, imm}:
#         limiter.apply(ctx, mesh, q)            # one or more adjusters
#
#     stepper.step(ctx, u, dt, compute_rhs, adjust)
#
# Both closures are alive across the `step` call, so they must not both
# capture the same object mutably: name `mut` only for what a closure
# actually mutates (the limiter's scratch buffers here) and leave the
# rest to the implicit read-only `imm` capture.
#
# For "no adjuster" (smooth problems, Maxwell, advection) pass
# `no_op_adjust[2, P, NC]`.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state import DGState


# ---------------------------------------------------------------------------
# Kernel: output[i] = x[i] + dt · rhs[i]
#
# Fused forward-Euler stage — only two inputs, written to a distinct output.
# Used by SSPRK3 stage 1 (where the "u_stage" and "u0" of the more general
# combine would alias each other and fall foul of Mojo's aliasing checker).
# ---------------------------------------------------------------------------

def _euler_step_kernel[
    XLayout: TensorLayout, RhsLayout: TensorLayout, OutLayout: TensorLayout,
](
    x: TileTensor[WXM_DTYPE, XLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    output: TileTensor[WXM_DTYPE, OutLayout, MutAnyOrigin],
    dt: Real,
    size_i64: Int64,
):
    var size = Int(size_i64)
    comptime assert x.flat_rank == 1, "euler_step expects 1D x"
    comptime assert rhs.flat_rank == 1, "euler_step expects 1D rhs"
    comptime assert output.flat_rank == 1, "euler_step expects 1D output"
    var t = global_idx.x
    if t < size:
        var xv = rebind[Scalar[WXM_DTYPE]](x[t])
        var rv = rebind[Scalar[WXM_DTYPE]](rhs[t])
        output[t] = rebind[output.ElementType](xv + dt * rv)


# ---------------------------------------------------------------------------
# Kernel: out[i] = a · u0[i] + b · ( u_stage[i] + dt · rhs[i] )
#
# Used by the SSPRK3 combination steps. `out` is allowed to alias `u_stage`.
# ---------------------------------------------------------------------------

def _ssprk3_combine_kernel[
    U0Layout: TensorLayout,
    USLayout: TensorLayout,
    RhsLayout: TensorLayout,
    OutLayout: TensorLayout,
](
    u0: TileTensor[WXM_DTYPE, U0Layout, MutAnyOrigin],
    u_stage: TileTensor[WXM_DTYPE, USLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    output: TileTensor[WXM_DTYPE, OutLayout, MutAnyOrigin],
    a: Real,
    b: Real,
    dt: Real,
    size_i64: Int64,
):
    var size = Int(size_i64)
    comptime assert u0.flat_rank == 1, "combine expects 1D u0"
    comptime assert u_stage.flat_rank == 1, "combine expects 1D u_stage"
    comptime assert rhs.flat_rank == 1, "combine expects 1D rhs"
    comptime assert output.flat_rank == 1, "combine expects 1D output"
    var t = global_idx.x
    if t < size:
        var u0v = rebind[Scalar[WXM_DTYPE]](u0[t])
        var usv = rebind[Scalar[WXM_DTYPE]](u_stage[t])
        var rhv = rebind[Scalar[WXM_DTYPE]](rhs[t])
        output[t] = rebind[output.ElementType](a * u0v + b * (usv + dt * rhv))


# DG-perf Task 4 attempt (2026-05-17) — `_ssprk3_combine_inplace_kernel`
# variant + skipping the `u_save ← u` copy was tried. The eliminated
# per-step D2D copy is ~0.2 % of step time at current mesh sizes, well
# below the 5 % target the task plan called for. The in-place kernel
# variant turned out to be _slower_ than the regular 4-arg combine
# (Mojo codegen difference, probably memory-fence-related on the
# aliased read/write), so 2D bench regressed by 1–4 %. Reverted.
#
# Recheck after mesh / NC scaling pushes the state buffer larger; for
# now Task 4 stays unfilled.


@always_inline
def no_op_adjust[NDIM: Int, P: Int, NC: Int](
    mut q: DGState[NDIM, P, NC],
) raises:
    """Identity adjuster — pass this when no limiter / positivity step is
    wanted (smooth problems, Maxwell, advection)."""
    pass


struct SSPRK3[NDIM: Int, P: Int, NC: Int]:
    """Generic SSPRK3 scratch + stepper for any NC-component DG problem in 1, 2 or 3D.

    The stepper itself owns only four DG-state scratch buffers; the
    spatial solver, mesh, and any adjusters are captured by closures
    passed to `step` at the call site.
    """

    var u_save: DGState[Self.NDIM, Self.P, Self.NC]
    var stage1: DGState[Self.NDIM, Self.P, Self.NC]
    var stage2: DGState[Self.NDIM, Self.P, Self.NC]
    var rhs: DGState[Self.NDIM, Self.P, Self.NC]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u_save = DGState[Self.NDIM, Self.P, Self.NC](ctx, num_elements)
        self.stage1 = DGState[Self.NDIM, Self.P, Self.NC](ctx, num_elements)
        self.stage2 = DGState[Self.NDIM, Self.P, Self.NC](ctx, num_elements)
        self.rhs = DGState[Self.NDIM, Self.P, Self.NC](ctx, num_elements)

    def step[
        ComputeRhs: def(
            mut DGState[Self.NDIM, Self.P, Self.NC],
            mut DGState[Self.NDIM, Self.P, Self.NC],
        ) raises -> None,
        Adjust: def(
            mut DGState[Self.NDIM, Self.P, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState[Self.NDIM, Self.P, Self.NC],
        dt: Real,
        compute_rhs_fn: ComputeRhs,
        adjust_fn: Adjust,
    ) raises:
        """One full SSPRK3 step on `u`.

        `compute_rhs_fn(q_in, q_out)` and `adjust_fn(q)` are user-supplied
        comptime closures (see module docstring for the pattern). The
        stepper handles only the three RK combines and the snapshot.
        """
        var size = u.buf_len
        comptime BLOCK = 256
        var grid = ceildiv(size, BLOCK)
        var layout = row_major(Int(size))

        adjust_fn(u)
        ctx.enqueue_copy(dst_buf=self.u_save.buf, src_buf=u.buf)

        # Stage 1: u^(1) = u^n + dt L(u^n)
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

        # Stage 2: u^(2) = 3/4 u^n + 1/4 (u^(1) + dt L(u^(1)))
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

        # Stage 3: u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt L(u^(2)))
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
            mut DGState[Self.NDIM, Self.P, Self.NC],
            mut DGState[Self.NDIM, Self.P, Self.NC],
        ) raises -> None,
        Adjust: def(
            mut DGState[Self.NDIM, Self.P, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState[Self.NDIM, Self.P, Self.NC],
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

    def run_until_cfl[
        ComputeRhs: def(
            mut DGState[Self.NDIM, Self.P, Self.NC],
            mut DGState[Self.NDIM, Self.P, Self.NC],
        ) raises -> None,
        Adjust: def(
            mut DGState[Self.NDIM, Self.P, Self.NC],
        ) raises -> None,
        SuggestDt: def(
            mut DGState[Self.NDIM, Self.P, Self.NC],
        ) raises -> Real,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState[Self.NDIM, Self.P, Self.NC],
        t: Real,
        t_target: Real,
        compute_rhs_fn: ComputeRhs,
        adjust_fn: Adjust,
        suggest_dt_fn: SuggestDt,
        dt_interval: Int = 1,
        dt_min: Real = 0.0,
    ) raises -> Real:
        """Advance from `t` to `t_target` with the step chosen by
        `suggest_dt_fn(u)`, typically a closure over
        `spatial.suggest_dt(ctx, mesh, q)`. Returns the time reached.

        The suggestion costs one GPU reduction and a host read-back, so it
        is refreshed only every `dt_interval` steps; raise that on smooth
        problems where the wave speed changes slowly.

        Raises if the suggestion is not finite and positive, or falls below
        `dt_min`. A step that keeps shrinking means the solution is
        degenerating (typically a node pinned on the density or pressure
        floor, whose sound speed then grows without bound); failing here
        is more useful than stalling.
        """
        var time = t
        var dt: Real = 0.0
        var since_refresh = dt_interval
        while time < t_target:
            if since_refresh >= dt_interval:
                dt = suggest_dt_fn(u)
                since_refresh = 0
                if not (dt > dt_min):
                    raise Error(
                        "run_until_cfl: suggested dt " + String(dt)
                        + " at t = " + String(time)
                        + " is not above dt_min = " + String(dt_min)
                    )
            var remaining = t_target - time
            # Take the remainder in one step if it is within round-off of dt.
            var dt_use = dt if dt < remaining * (Real(1.0) - Real(1.0e-12)) else remaining
            self.step(ctx, u, dt_use, compute_rhs_fn, adjust_fn)
            since_refresh += 1
            if dt_use == remaining:
                return t_target
            time += dt_use
        return time
