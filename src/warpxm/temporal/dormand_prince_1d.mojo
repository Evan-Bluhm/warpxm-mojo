# Dormand-Prince 5(4) embedded explicit Runge-Kutta stepper.
#
# Provides both fixed-dt `step` and adaptive `step_adaptive` on a
# `DormandPrince45_1D[P, NC]` struct. The 2D and 3D wrappers
# (`dormand_prince_{2,3}d.mojo`) import the dimension-agnostic combine
# kernels (`_combine{1..5}`), per-cell error kernel
# (`_dp45_error_kernel`), and single-block reduction kernel
# (`_dp45_sum_sq_kernel`) from this file unchanged — the buffers are
# flat in every dimension, so the kernels don't care.
#
# Standard DP45 Butcher tableau (Dormand & Prince 1980, JCAM 6(1)):
#
#   c2 = 1/5
#   c3 = 3/10
#   c4 = 4/5
#   c5 = 8/9
#   c6 = 1
#   c7 = 1
#
#   a21 = 1/5
#   a31 = 3/40,     a32 = 9/40
#   a41 = 44/45,    a42 = -56/15,    a43 = 32/9
#   a51 = 19372/6561,  a52 = -25360/2187,  a53 = 64448/6561,  a54 = -212/729
#   a61 = 9017/3168,   a62 = -355/33,      a63 = 46732/5247,
#   a64 = 49/176,      a65 = -5103/18656
#
#   b₅ = (35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0)
#
# The embedded 4th-order coefficients enter only via
# `_dp45_error_kernel` (which writes the y_5 - y_4 difference); the
# adaptive controller is in `step_adaptive`.

from max.gpu import barrier, global_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from std.math import ceildiv

from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state_1d import DGState1D


comptime _DP45_REDUCE_BLOCK: Int = 256
"""Block size for the single-block error-norm reduction kernel. Must be a
power of two; the kernel assumes `block_dim.x == _DP45_REDUCE_BLOCK`."""


# Per-stage combine kernels. Each takes exactly the right number of `k`
# buffers — Mojo's aliasing checker forbids passing the same buffer to two
# `MutAnyOrigin` slots, so we cannot use a single 7-input kernel.

def _combine1[L: TensorLayout](
    y0: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k1: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    y_out: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    c1: Real, dt: Real, size_i64: Int64,
):
    var size = Int(size_i64)
    comptime assert y0.flat_rank == 1
    var t = global_idx.x
    if t < size:
        var y0v = rebind[Scalar[WXM_DTYPE]](y0[t])
        var s = c1 * rebind[Scalar[WXM_DTYPE]](k1[t])
        y_out[t] = rebind[y_out.ElementType](y0v + dt * s)


def _combine2[L: TensorLayout](
    y0: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k1: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k2: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    y_out: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    c1: Real, c2: Real, dt: Real, size_i64: Int64,
):
    var size = Int(size_i64)
    comptime assert y0.flat_rank == 1
    var t = global_idx.x
    if t < size:
        var y0v = rebind[Scalar[WXM_DTYPE]](y0[t])
        var s = c1 * rebind[Scalar[WXM_DTYPE]](k1[t])
        s += c2 * rebind[Scalar[WXM_DTYPE]](k2[t])
        y_out[t] = rebind[y_out.ElementType](y0v + dt * s)


def _combine3[L: TensorLayout](
    y0: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k1: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k2: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k3: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    y_out: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    c1: Real, c2: Real, c3: Real, dt: Real, size_i64: Int64,
):
    var size = Int(size_i64)
    comptime assert y0.flat_rank == 1
    var t = global_idx.x
    if t < size:
        var y0v = rebind[Scalar[WXM_DTYPE]](y0[t])
        var s = c1 * rebind[Scalar[WXM_DTYPE]](k1[t])
        s += c2 * rebind[Scalar[WXM_DTYPE]](k2[t])
        s += c3 * rebind[Scalar[WXM_DTYPE]](k3[t])
        y_out[t] = rebind[y_out.ElementType](y0v + dt * s)


def _combine4[L: TensorLayout](
    y0: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k1: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k2: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k3: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k4: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    y_out: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    c1: Real, c2: Real, c3: Real, c4: Real, dt: Real, size_i64: Int64,
):
    var size = Int(size_i64)
    comptime assert y0.flat_rank == 1
    var t = global_idx.x
    if t < size:
        var y0v = rebind[Scalar[WXM_DTYPE]](y0[t])
        var s = c1 * rebind[Scalar[WXM_DTYPE]](k1[t])
        s += c2 * rebind[Scalar[WXM_DTYPE]](k2[t])
        s += c3 * rebind[Scalar[WXM_DTYPE]](k3[t])
        s += c4 * rebind[Scalar[WXM_DTYPE]](k4[t])
        y_out[t] = rebind[y_out.ElementType](y0v + dt * s)


def _dp45_error_kernel[L: TensorLayout](
    k1: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k3: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k4: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k5: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k6: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k7: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    err: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    dt: Real, size_i64: Int64,
):
    """Write `err[i] = dt * (e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)`,
    the y_5 - y_4 difference. e_i = b5_i - b4_i (e2 = 0)."""
    var size = Int(size_i64)
    comptime assert k1.flat_rank == 1
    var t = global_idx.x
    if t < size:
        var s = Real(71.0 / 57600.0) * rebind[Scalar[WXM_DTYPE]](k1[t])
        s += Real(-71.0 / 16695.0) * rebind[Scalar[WXM_DTYPE]](k3[t])
        s += Real(71.0 / 1920.0) * rebind[Scalar[WXM_DTYPE]](k4[t])
        s += Real(-17253.0 / 339200.0) * rebind[Scalar[WXM_DTYPE]](k5[t])
        s += Real(22.0 / 525.0) * rebind[Scalar[WXM_DTYPE]](k6[t])
        s += Real(-1.0 / 40.0) * rebind[Scalar[WXM_DTYPE]](k7[t])
        err[t] = rebind[err.ElementType](dt * s)


def _dp45_sum_sq_kernel[
    EL: TensorLayout, OL: TensorLayout,
](
    err: TileTensor[WXM_DTYPE, EL, MutAnyOrigin],
    u_save: TileTensor[WXM_DTYPE, EL, MutAnyOrigin],
    u_new: TileTensor[WXM_DTYPE, EL, MutAnyOrigin],
    out_sum_sq: TileTensor[WXM_DTYPE, OL, MutAnyOrigin],
    atol: Real, rtol: Real, size_i64: Int64,
):
    """Single-block reduction: writes Σ_i (err_i / sc_i)^2 to out_sum_sq[0],
    where sc_i = atol + rtol · max(|u_save_i|, |u_new_i|). Launch with
    grid_dim=1, block_dim=_DP45_REDUCE_BLOCK."""
    var size = Int(size_i64)
    comptime assert err.flat_rank == 1
    comptime assert out_sum_sq.flat_rank == 1
    var sums = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[_DP45_REDUCE_BLOCK]())

    var tid = Int(thread_idx.x)
    var local = Real(0.0)
    var i = tid
    while i < size:
        var e = rebind[Scalar[WXM_DTYPE]](err[i])
        var u_n = rebind[Scalar[WXM_DTYPE]](u_save[i])
        var u_v = rebind[Scalar[WXM_DTYPE]](u_new[i])
        var au_n = u_n if u_n >= Real(0.0) else -u_n
        var au_v = u_v if u_v >= Real(0.0) else -u_v
        var au_max = au_n if au_n > au_v else au_v
        var sc = atol + rtol * au_max
        var r = e / sc
        local += r * r
        i += _DP45_REDUCE_BLOCK
    sums[tid] = rebind[sums.ElementType](local)
    barrier()

    # Tree reduce in shared memory (BLOCK is a power of 2).
    var active = _DP45_REDUCE_BLOCK
    while active > 1:
        active >>= 1
        if tid < active:
            var l = rebind[Scalar[WXM_DTYPE]](sums[tid])
            var r2 = rebind[Scalar[WXM_DTYPE]](sums[tid + active])
            sums[tid] = rebind[sums.ElementType](l + r2)
        barrier()

    if tid == 0:
        out_sum_sq[0] = rebind[out_sum_sq.ElementType](
            rebind[Scalar[WXM_DTYPE]](sums[0])
        )


def _combine5[L: TensorLayout](
    y0: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k1: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k2: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k3: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k4: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    k5: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    y_out: TileTensor[WXM_DTYPE, L, MutAnyOrigin],
    c1: Real, c2: Real, c3: Real, c4: Real, c5: Real,
    dt: Real, size_i64: Int64,
):
    var size = Int(size_i64)
    comptime assert y0.flat_rank == 1
    var t = global_idx.x
    if t < size:
        var y0v = rebind[Scalar[WXM_DTYPE]](y0[t])
        var s = c1 * rebind[Scalar[WXM_DTYPE]](k1[t])
        s += c2 * rebind[Scalar[WXM_DTYPE]](k2[t])
        s += c3 * rebind[Scalar[WXM_DTYPE]](k3[t])
        s += c4 * rebind[Scalar[WXM_DTYPE]](k4[t])
        s += c5 * rebind[Scalar[WXM_DTYPE]](k5[t])
        y_out[t] = rebind[y_out.ElementType](y0v + dt * s)


struct DormandPrince45_1D[P: Int, NC: Int]:
    """Device-resident DP45 integrator for 1D DG fields. Provides fixed-step
    `step` and adaptive `step_adaptive`."""

    var u_save: DGState1D[Self.P, Self.NC]
    var y_stage: DGState1D[Self.P, Self.NC]
    var k1: DGState1D[Self.P, Self.NC]
    var k2: DGState1D[Self.P, Self.NC]
    var k3: DGState1D[Self.P, Self.NC]
    var k4: DGState1D[Self.P, Self.NC]
    var k5: DGState1D[Self.P, Self.NC]
    var k6: DGState1D[Self.P, Self.NC]
    var k7: DGState1D[Self.P, Self.NC]
    """Extra k7 buffer used by `step_adaptive` for the embedded 4th-order
    error estimate. Unused by the fixed-step `step`."""
    var err_buf: DGState1D[Self.P, Self.NC]
    """Per-cell (y_5 - y_4) difference vector used for adaptive control."""
    var sum_sq_buf: DeviceBuffer[Real.dtype]
    """One-element scratch buffer: target of the GPU-side error-norm reduction
    in `step_adaptive`. Avoids three full-buffer `map_to_host` copies per step."""

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.u_save = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.y_stage = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.k1 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.k2 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.k3 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.k4 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.k5 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.k6 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.k7 = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.err_buf = DGState1D[Self.P, Self.NC](ctx, num_elements)
        self.sum_sq_buf = ctx.enqueue_create_buffer[Real.dtype](1)

    def step[
        ComputeRhs: def(
            mut DGState1D[Self.P, Self.NC],
            mut DGState1D[Self.P, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState1D[Self.P, Self.NC],
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

        # k1 = f(u^n)
        compute_rhs_fn(self.u_save, self.k1)

        # k2 = f(u + dt*(1/5)*k1)
        ctx.enqueue_function[k_c1](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(1.0) / Real(5.0), dt, Int64(size),
            grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k2)

        # k3 = f(u + dt*(3/40*k1 + 9/40*k2))
        ctx.enqueue_function[k_c2](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.k2.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(3.0) / Real(40.0),
            Real(9.0) / Real(40.0),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k3)

        # k4 = f(u + dt*(44/45*k1 - 56/15*k2 + 32/9*k3))
        ctx.enqueue_function[k_c3](
            TileTensor(self.u_save.buf, lay),
            TileTensor(self.k1.buf, lay),
            TileTensor(self.k2.buf, lay),
            TileTensor(self.k3.buf, lay),
            TileTensor(self.y_stage.buf, lay),
            Real(44.0) / Real(45.0),
            -Real(56.0) / Real(15.0),
            Real(32.0) / Real(9.0),
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k4)

        # k5 = f(u + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
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
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k5)

        # k6 = f(u + dt*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))
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
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )
        compute_rhs_fn(self.y_stage, self.k6)

        # u^{n+1} = u + dt*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
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
            dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
        )

    def step_adaptive[
        ComputeRhs: def(
            mut DGState1D[Self.P, Self.NC],
            mut DGState1D[Self.P, Self.NC],
        ) raises -> None,
    ](
        mut self,
        var ctx: DeviceContext,
        mut u: DGState1D[Self.P, Self.NC],
        dt_init: Real,
        compute_rhs_fn: ComputeRhs,
        atol: Real = Real(1.0e-6),
        rtol: Real = Real(1.0e-3),
        max_attempts: Int = 10,
    ) raises -> Tuple[Real, Real]:
        """Take one adaptive DP45 step. Returns `(dt_used, dt_next)`:
        `dt_used` is the dt actually taken (≤ dt_init if shrunk on reject);
        `dt_next` is the suggested initial dt for the *next* call.

        atol / rtol form the per-component error scale
            sc_i = atol + rtol · max(|u_n_i|, |y_5_i|),
        and the controller accepts when the RMS scaled error err < 1.
        """
        comptime BLOCK = 256
        var size = u.buf_len
        var grid = ceildiv(size, BLOCK)
        var lay = row_major(Int(size))
        var lay1 = row_major[1]()
        comptime k_err = _dp45_error_kernel[type_of(lay)]
        comptime k_sum = _dp45_sum_sq_kernel[type_of(lay), type_of(lay1)]

        # Snapshot u^n once; if a trial is rejected we restore from this.
        ctx.enqueue_copy(dst_buf=self.u_save.buf, src_buf=u.buf)

        var dt = dt_init
        var attempt = 0
        while True:
            # On retry, u was already restored from u_save at the bottom of
            # the previous loop iteration. self.step writes a fresh u^{n+1}
            # using u_save (snapshot) as the starting point.
            ctx.enqueue_copy(dst_buf=u.buf, src_buf=self.u_save.buf)
            self.step(ctx, u, dt, compute_rhs_fn)

            # k7 = f(y_5)
            compute_rhs_fn(u, self.k7)
            ctx.enqueue_function[k_err](
                TileTensor(self.k1.buf, lay),
                TileTensor(self.k3.buf, lay),
                TileTensor(self.k4.buf, lay),
                TileTensor(self.k5.buf, lay),
                TileTensor(self.k6.buf, lay),
                TileTensor(self.k7.buf, lay),
                TileTensor(self.err_buf.buf, lay),
                dt, Int64(size), grid_dim=grid, block_dim=BLOCK,
            )

            # GPU-side reduction: one block tree-reduces Σ (err/sc)^2 into
            # sum_sq_buf[0]. Avoids three full-buffer device→host copies that
            # were dominating cost on large meshes.
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

            # Standard PI-free controller: dt_new = dt · 0.9 · err^(-1/5),
            # clipped to [0.2·dt, 5·dt].
            var safety = Real(0.9)
            var exponent = -Real(0.2)
            var factor = safety * (err if err > Real(1.0e-30) else Real(1.0e-30)) ** exponent
            if factor < Real(0.2): factor = 0.2
            if factor > Real(5.0): factor = 5.0
            var dt_next = dt * factor

            if err <= Real(1.0):
                # Accept.
                return Tuple[Real, Real](dt, dt_next)

            # Reject. Restore u from u_save and retry with smaller dt.
            attempt += 1
            if attempt >= max_attempts:
                # Give up: accept the current step even though err > 1 (avoid
                # spinning forever). Caller can decide whether to bail.
                return Tuple[Real, Real](dt, dt_next)
            dt = dt_next
