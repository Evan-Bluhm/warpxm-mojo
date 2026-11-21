# Generic SSPRK3 (three-stage Shu-Osher strong-stability-preserving
# Runge-Kutta) stepper for 2D DG fields, driven by two closures that
# the caller passes as trailing arguments:
#
#   * `compute_rhs_fn(q_in, q_out)` — evaluates the spatial DG residual
#     for the supplied input state and writes it to `q_out`. Captures
#     whichever spatial solver / mesh / device context the driver is
#     using.
#   * `adjust_fn(q)` — runs after each stage value is written; calls
#     limiter / positivity / nothing as the problem demands.
#
# This collapses the previous matrix of `SSPRK3_Euler_Limited_2D`,
# `SSPRK3_Euler_Positive_2D`, `SSPRK3_IMHD_Limited_2D`,
# `SSPRK3_IMHD_Positive_2D`, `SSPRK3_IMHD_GLM_Limited_2D` and
# `SSPRK3_Maxwell_2D` files into one stepper. Adjuster composition is
# done at the call site rather than baked into a per-physics stepper.
#
# Pattern (driver code):
#
#     var spatial = DGEulerSolver2D[P](app)
#     var limiter = MoeRossmanith2D[P, EULER_NC](ctx, num_elements)
#     var stepper = SSPRK3_2D[P, EULER_NC](ctx, num_elements)
#
#     def compute_rhs(
#         mut q_in: DGState2D[P, EULER_NC],
#         mut q_out: DGState2D[P, EULER_NC],
#     ) raises {imm}:
#         spatial.compute_rhs(ctx, mesh, q_in, q_out)
#
#     def adjust(mut q: DGState2D[P, EULER_NC]) raises {mut limiter, imm}:
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
# `no_op_adjust_2d[P, NC]`.

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.math import ceildiv

from warpxm.core.types import Real
from warpxm.temporal.ssprk3 import _euler_step_kernel, _ssprk3_combine_kernel
from warpxm.variable.state_2d import DGState2D


@always_inline
def no_op_adjust_2d[P: Int, NC: Int](
    mut q: DGState2D[P, NC],
) raises:
    """Identity adjuster — pass this when no limiter / positivity step is
    wanted (smooth problems, Maxwell, advection)."""
    pass


struct SSPRK3_2D[P: Int, NC: Int]:
    """Generic SSPRK3 scratch + stepper for any 2D NC-component problem.

    The stepper itself owns only four DG-state scratch buffers; the
    spatial solver, mesh, and any adjusters are captured by closures
    passed to `step` at the call site.
    """

    var u_save: DGState2D[Self.P, Self.NC]
    var stage1: DGState2D[Self.P, Self.NC]
    var stage2: DGState2D[Self.P, Self.NC]
    var rhs: DGState2D[Self.P, Self.NC]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u_save = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.stage1 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.stage2 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.rhs = DGState2D[Self.P, Self.NC](ctx, num_elements)

    def step[
        ComputeRhs: def(
            mut DGState2D[Self.P, Self.NC],
            mut DGState2D[Self.P, Self.NC],
        ) raises -> None,
        Adjust: def(
            mut DGState2D[Self.P, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState2D[Self.P, Self.NC],
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
