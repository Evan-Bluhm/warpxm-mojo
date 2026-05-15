# Dormand-Prince 5(4) stepper for 2D Euler (fixed-step + adaptive).
#
# Direct extension of `temporal/dormand_prince_1d.mojo` to 2D — the combine
# and error kernels operate on flat buffers and are reused unchanged. Only
# the host-side driver changes (DGState2D / BlockMesh2D / DGEulerSolver2D
# in place of their 1D analogues).

from std.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, Idx, row_major
from std.math import ceildiv

from warpxm.apps.euler_2d import EULER_NC
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.dormand_prince_1d import (
    _combine1, _combine2, _combine3, _combine4, _combine5, _dp45_error_kernel,
)
from warpxm.variable.state_2d import DGState2D


struct DormandPrince45_Euler_2D[P: Int]:
    """Device-resident DP45 integrator for 2D Euler fields."""

    var u_save: DGState2D[Self.P, EULER_NC]
    var y_stage: DGState2D[Self.P, EULER_NC]
    var k1: DGState2D[Self.P, EULER_NC]
    var k2: DGState2D[Self.P, EULER_NC]
    var k3: DGState2D[Self.P, EULER_NC]
    var k4: DGState2D[Self.P, EULER_NC]
    var k5: DGState2D[Self.P, EULER_NC]
    var k6: DGState2D[Self.P, EULER_NC]
    var k7: DGState2D[Self.P, EULER_NC]
    var err_buf: DGState2D[Self.P, EULER_NC]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u_save = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.y_stage = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.k1 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.k2 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.k3 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.k4 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.k5 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.k6 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.k7 = DGState2D[Self.P, EULER_NC](ctx, num_elements)
        self.err_buf = DGState2D[Self.P, EULER_NC](ctx, num_elements)

    def step(
        mut self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        spatial: DGEulerSolver2D[Self.P],
        mut u: DGState2D[Self.P, EULER_NC],
        dt: Real,
    ) raises:
        var size = u.buf_len
        comptime BLOCK = 256
        var grid = ceildiv(size, BLOCK)
        var lay = row_major(Idx(size))
        comptime k_c1 = _combine1[type_of(lay)]
        comptime k_c2 = _combine2[type_of(lay)]
        comptime k_c3 = _combine3[type_of(lay)]
        comptime k_c4 = _combine4[type_of(lay)]
        comptime k_c5 = _combine5[type_of(lay)]

        ctx.enqueue_copy(dst_buf=self.u_save.buf, src_buf=u.buf)

        spatial.compute_rhs(ctx, mesh, self.u_save, self.k1)

        ctx.enqueue_function[k_c1](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(1.0) / Real(5.0), dt, size,
            grid_dim=grid, block_dim=BLOCK,
        )
        spatial.compute_rhs(ctx, mesh, self.y_stage, self.k2)

        ctx.enqueue_function[k_c2](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.k2.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(3.0) / Real(40.0),
            Real(9.0) / Real(40.0),
            dt, size, grid_dim=grid, block_dim=BLOCK,
        )
        spatial.compute_rhs(ctx, mesh, self.y_stage, self.k3)

        ctx.enqueue_function[k_c3](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.k2.buf, lay),
            TileTensor(self.k3.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(44.0) / Real(45.0),
            -Real(56.0) / Real(15.0),
            Real(32.0) / Real(9.0),
            dt, size, grid_dim=grid, block_dim=BLOCK,
        )
        spatial.compute_rhs(ctx, mesh, self.y_stage, self.k4)

        ctx.enqueue_function[k_c4](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.k2.buf, lay),
            TileTensor(self.k3.buf, lay),
            TileTensor(self.k4.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(19372.0) / Real(6561.0),
            -Real(25360.0) / Real(2187.0),
            Real(64448.0) / Real(6561.0),
            -Real(212.0) / Real(729.0),
            dt, size, grid_dim=grid, block_dim=BLOCK,
        )
        spatial.compute_rhs(ctx, mesh, self.y_stage, self.k5)

        ctx.enqueue_function[k_c5](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.k2.buf, lay),
            TileTensor(self.k3.buf, lay),
            TileTensor(self.k4.buf, lay),
            TileTensor(self.k5.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(9017.0) / Real(3168.0),
            -Real(355.0) / Real(33.0),
            Real(46732.0) / Real(5247.0),
            Real(49.0) / Real(176.0),
            -Real(5103.0) / Real(18656.0),
            dt, size, grid_dim=grid, block_dim=BLOCK,
        )
        spatial.compute_rhs(ctx, mesh, self.y_stage, self.k6)

        ctx.enqueue_function[k_c5](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.k3.buf, lay),
            TileTensor(self.k4.buf, lay),
            TileTensor(self.k5.buf, lay),
            TileTensor(self.k6.buf, lay),
            TileTensor(u.buf, lay),
            Real(35.0) / Real(384.0),
            Real(500.0) / Real(1113.0),
            Real(125.0) / Real(192.0),
            -Real(2187.0) / Real(6784.0),
            Real(11.0) / Real(84.0),
            dt, size, grid_dim=grid, block_dim=BLOCK,
        )

    def step_adaptive(
        mut self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        spatial: DGEulerSolver2D[Self.P],
        mut u: DGState2D[Self.P, EULER_NC],
        dt_init: Real,
        atol: Real = Real(1.0e-6),
        rtol: Real = Real(1.0e-3),
        max_attempts: Int = 10,
    ) raises -> Tuple[Real, Real]:
        """One adaptive DP45 step on a 2D Euler state. Returns
        `(dt_used, dt_next)`. See `dormand_prince_1d.mojo` for the controller
        description.
        """
        comptime BLOCK = 256
        var size = u.buf_len
        var grid = ceildiv(size, BLOCK)
        var lay = row_major(Idx(size))
        comptime k_err = _dp45_error_kernel[type_of(lay)]

        ctx.enqueue_copy(dst_buf=self.u_save.buf, src_buf=u.buf)

        var dt = dt_init
        var attempt = 0
        while True:
            ctx.enqueue_copy(dst_buf=u.buf, src_buf=self.u_save.buf)
            self.step(ctx, mesh, spatial, u, dt)

            spatial.compute_rhs(ctx, mesh, u, self.k7)
            ctx.enqueue_function[k_err](
                TileTensor(self.k1.buf, lay),
                TileTensor(self.k3.buf, lay),
                TileTensor(self.k4.buf, lay),
                TileTensor(self.k5.buf, lay),
                TileTensor(self.k6.buf, lay),
                TileTensor(self.k7.buf, lay),
                TileTensor(self.err_buf.buf, lay),
                dt, size, grid_dim=grid, block_dim=BLOCK,
            )
            ctx.synchronize()

            var sum_sq: Real = 0.0
            with self.err_buf.buf.map_to_host() as eh:
                with u.buf.map_to_host() as uh:
                    with self.u_save.buf.map_to_host() as u0h:
                        for i in range(size):
                            var u_n = u0h[i]
                            var u_new = uh[i]
                            var au_n = u_n if u_n >= Real(0.0) else -u_n
                            var au_new = u_new if u_new >= Real(0.0) else -u_new
                            var au_max = au_n if au_n > au_new else au_new
                            var sc = atol + rtol * au_max
                            var r = eh[i] / sc
                            sum_sq += r * r
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
