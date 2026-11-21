# Generic Dormand-Prince 5(4) stepper for 2D DG fields, NC-parameterised.
#
# Same Butcher tableau as `dormand_prince_1d.mojo`; the combine kernels
# (`_combine{1..5}`), per-cell error kernel (`_dp45_error_kernel`), and
# single-block error-norm reduction kernel (`_dp45_sum_sq_kernel`) all
# live there and are imported here unchanged — they operate on flat
# buffers and don't care about dimensionality. The driver supplies
# `compute_rhs_fn` as a trailing closure argument, exactly like the
# SSPRK3 family.

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major
from std.math import ceildiv

from warpxm.core.types import Real
from warpxm.temporal.dormand_prince_1d import (
    _combine1, _combine2, _combine3, _combine4, _combine5, _dp45_error_kernel,
    _dp45_sum_sq_kernel, _DP45_REDUCE_BLOCK,
)
from warpxm.variable.state_2d import DGState2D


struct DormandPrince45_2D[P: Int, NC: Int]:
    """Generic DP45 integrator for any 2D NC-component DG problem."""

    var u_save: DGState2D[Self.P, Self.NC]
    var y_stage: DGState2D[Self.P, Self.NC]
    var k1: DGState2D[Self.P, Self.NC]
    var k2: DGState2D[Self.P, Self.NC]
    var k3: DGState2D[Self.P, Self.NC]
    var k4: DGState2D[Self.P, Self.NC]
    var k5: DGState2D[Self.P, Self.NC]
    var k6: DGState2D[Self.P, Self.NC]
    var k7: DGState2D[Self.P, Self.NC]
    var err_buf: DGState2D[Self.P, Self.NC]
    var sum_sq_buf: DeviceBuffer[Real.dtype]
    """One-element scratch buffer for the GPU-side error-norm reduction."""

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u_save = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.y_stage = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.k1 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.k2 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.k3 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.k4 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.k5 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.k6 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.k7 = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.err_buf = DGState2D[Self.P, Self.NC](ctx, num_elements)
        self.sum_sq_buf = ctx.enqueue_create_buffer[Real.dtype](1)

    def step[
        ComputeRhs: def(
            mut DGState2D[Self.P, Self.NC],
            mut DGState2D[Self.P, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState2D[Self.P, Self.NC],
        dt: Real,
        compute_rhs_fn: ComputeRhs,
    ) raises:
        var size = u.buf_len
        comptime BLOCK = 256
        var grid = ceildiv(size, BLOCK)
        var lay = row_major(Int(size))
        comptime k_c1 = _combine1[type_of(lay)]
        comptime k_c2 = _combine2[type_of(lay)]
        comptime k_c3 = _combine3[type_of(lay)]
        comptime k_c4 = _combine4[type_of(lay)]
        comptime k_c5 = _combine5[type_of(lay)]

        ctx.enqueue_copy(dst_buf=self.u_save.buf, src_buf=u.buf)
        compute_rhs_fn(self.u_save, self.k1)

        ctx.enqueue_function[k_c1](
            TileTensor(self.u_save.buf, lay), TileTensor(self.k1.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(1.0) / Real(5.0), dt, Int64(size),
            grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k2)

        ctx.enqueue_function[k_c2](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay), TileTensor(self.k2.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(3.0) / Real(40.0), Real(9.0) / Real(40.0),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k3)

        ctx.enqueue_function[k_c3](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay), TileTensor(self.k2.buf, lay),
            TileTensor(self.k3.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(44.0) / Real(45.0), -Real(56.0) / Real(15.0), Real(32.0) / Real(9.0),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k4)

        ctx.enqueue_function[k_c4](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay), TileTensor(self.k2.buf, lay),
            TileTensor(self.k3.buf, lay), TileTensor(self.k4.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(19372.0) / Real(6561.0),
            -Real(25360.0) / Real(2187.0),
            Real(64448.0) / Real(6561.0),
            -Real(212.0) / Real(729.0),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k5)

        ctx.enqueue_function[k_c5](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay), TileTensor(self.k2.buf, lay),
            TileTensor(self.k3.buf, lay), TileTensor(self.k4.buf, lay),
            TileTensor(self.k5.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(9017.0) / Real(3168.0),
            -Real(355.0) / Real(33.0),
            Real(46732.0) / Real(5247.0),
            Real(49.0) / Real(176.0),
            -Real(5103.0) / Real(18656.0),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k6)

        # u^{n+1} = u + dt*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
        ctx.enqueue_function[k_c5](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay), TileTensor(self.k3.buf, lay),
            TileTensor(self.k4.buf, lay), TileTensor(self.k5.buf, lay),
            TileTensor(self.k6.buf, lay),
            TileTensor(u.buf, lay),
            Real(35.0) / Real(384.0),
            Real(500.0) / Real(1113.0),
            Real(125.0) / Real(192.0),
            -Real(2187.0) / Real(6784.0),
            Real(11.0) / Real(84.0),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )

    def step_adaptive[
        ComputeRhs: def(
            mut DGState2D[Self.P, Self.NC],
            mut DGState2D[Self.P, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState2D[Self.P, Self.NC],
        dt_init: Real,
        compute_rhs_fn: ComputeRhs,
        atol: Real = Real(1.0e-6),
        rtol: Real = Real(1.0e-3),
        max_attempts: Int = 10,
    ) raises -> Tuple[Real, Real]:
        """One adaptive DP45 step. See `dormand_prince_1d.mojo` for the
        controller description."""
        comptime BLOCK = 256
        var size = u.buf_len
        var grid = ceildiv(size, BLOCK)
        var lay = row_major(Int(size))
        var lay1 = row_major[1]()
        comptime k_err = _dp45_error_kernel[type_of(lay)]
        comptime k_sum = _dp45_sum_sq_kernel[type_of(lay), type_of(lay1)]

        ctx.enqueue_copy(dst_buf=self.u_save.buf, src_buf=u.buf)

        var dt = dt_init
        var attempt = 0
        while True:
            ctx.enqueue_copy(dst_buf=u.buf, src_buf=self.u_save.buf)
            self.step(ctx, u, dt, compute_rhs_fn)
            compute_rhs_fn(u, self.k7)
            ctx.enqueue_function[k_err](
                TileTensor(self.k1.buf, lay), TileTensor(self.k3.buf, lay),
                TileTensor(self.k4.buf, lay), TileTensor(self.k5.buf, lay),
                TileTensor(self.k6.buf, lay), TileTensor(self.k7.buf, lay),
                TileTensor(self.err_buf.buf, lay),
                dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
            )
            ctx.enqueue_function[k_sum](
                TileTensor(self.err_buf.buf, lay),
                TileTensor(self.u_save.buf, lay),
                TileTensor(u.buf, lay),
                TileTensor(self.sum_sq_buf, lay1),
                atol, rtol, Int64(size),
                grid_dim=1, block_dim=_DP45_REDUCE_BLOCK,
            )
            var sum_sq: Real
            with self.sum_sq_buf.map_to_host() as sh:
                sum_sq = sh[0]
            var err = (sum_sq / Real(size)) ** Real(0.5)

            var safety = Real(0.9)
            var exponent = -Real(0.2)
            var factor = safety * (err if err > Real(1.0e-30) else Real(1.0e-30)) ** exponent
            if factor < Real(0.2): factor = 0.2
            if factor > Real(5.0): factor = 5.0
            var dt_next = dt * factor

            if err <= Real(1.0):
                return Tuple[Real, Real](dt, dt_next)

            attempt += 1
            if attempt >= max_attempts:
                return Tuple[Real, Real](dt, dt_next)
            dt = dt_next
