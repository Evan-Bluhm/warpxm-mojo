# CFL time-step selection.
#
# An explicit nodal DG scheme of degree P on a uniform mesh is stable for
#
#     Δt ≤ cfl / ((2P + 1) · Σ_d λ_d / Δx_d)
#
# where λ_d bounds the signal speed along axis d. For the nonlinear
# systems we take one direction-independent bound λ (|u| + c for Euler,
# |u| + the perpendicular fast speed for MHD) and reduce its maximum over
# all nodes on the GPU, so only one scalar reaches the host.
#
# Every DG solver exposes this as `solver.suggest_dt(ctx, mesh, q)`, using
# its app's `cfl_max`. With SSPRK3 on the GLL-collocation scheme used here,
# linear advection is stable up to `cfl_max` of about 1.5 for P = 1..4 and
# unstable by 2 for P >= 3 (`tests/test_cfl_timestep.mojo` pins this). Use
# 0.5 for problems with shocks, where the limiter and the direction-
# independent speed bound both want margin.

from std.collections import Array
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation

from warpxm.apps.euler import Euler, EULER_NC
from warpxm.apps.imhd import IMHD, MHD_NC
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC
from warpxm.core.types import Real, WXM_DTYPE, WXM_TINY, WXM_INF
from warpxm.spatial.dg_launch import KernelParams, RealTensor, kernel_params
from warpxm.variable.state import DGState


comptime _REDUCE_BLOCK: Int = 256
"""Threads in the single reduction block. Must be a power of two."""

comptime SPEED_EULER: Int = 0
comptime SPEED_IMHD: Int = 1
comptime SPEED_IMHD_GLM: Int = 2


def cfl_dt[P: Int](cfl_max: Real, speed_over_dx: Real) -> Real:
    """Stable step for `Σ_d λ_d / Δx_d = speed_over_dx`."""
    if speed_over_dx < WXM_TINY:
        return WXM_INF
    return cfl_max / (Real(2 * P + 1) * speed_over_dx)


def _max_speed_kernel[NC: Int, KIND: Int](
    q: RealTensor,
    out_max: RealTensor,
    num_nodes_i64: Int64,
    params: KernelParams,
):
    """Single-block reduction: writes max over nodes of the signal speed to
    `out_max[0]`. Launch with grid_dim=1, block_dim=_REDUCE_BLOCK."""
    var num_nodes = Int(num_nodes_i64)
    var gamma = params[0]
    var rho_min = params[1]
    var press_min = params[2]
    var c_h = params[3]
    var partial = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[_REDUCE_BLOCK]())

    var tid = Int(thread_idx.x)
    var local = Real(0.0)
    var n = tid
    while n < num_nodes:
        var qn = Array[Real, NC](fill=Real(0.0))
        comptime for c in range(NC):
            qn[c] = rebind[Real](q[n * NC + c])
        var s: Real
        comptime match KIND:
            case 0:
                s = Euler.max_signal_speed_static(
                    rebind[Array[Real, EULER_NC]](qn), gamma, rho_min, press_min,
                )
            case 1:
                s = IMHD.max_signal_speed_static(
                    rebind[Array[Real, MHD_NC]](qn), gamma, rho_min, press_min,
                )
            case _:
                s = ImhdGLM.max_signal_speed_static(
                    rebind[Array[Real, MHD_GLM_NC]](qn),
                    gamma, rho_min, press_min, c_h,
                )
        if s > local:
            local = s
        n += _REDUCE_BLOCK
    partial[tid] = rebind[partial.ElementType](local)
    barrier()

    var active = _REDUCE_BLOCK
    while active > 1:
        active >>= 1
        if tid < active:
            var a = rebind[Real](partial[tid])
            var b = rebind[Real](partial[tid + active])
            if b > a:
                partial[tid] = rebind[partial.ElementType](b)
        barrier()

    if tid == 0:
        out_max[0] = rebind[out_max.ElementType](rebind[Real](partial[0]))


def _max_speed[NDIM: Int, P: Int, NC: Int, KIND: Int](
    var ctx: DeviceContext,
    mut q: DGState[NDIM, P, NC],
    params: KernelParams,
) raises -> Real:
    var out_buf = ctx.enqueue_create_buffer[Real.dtype](1)
    ctx.enqueue_function[_max_speed_kernel[NC, KIND]](
        TileTensor(q.buf, row_major(Int(q.buf_len))),
        TileTensor(out_buf, row_major(Int(1))),
        Int64(q.total_nodes), params,
        grid_dim=1, block_dim=_REDUCE_BLOCK,
    )
    var speed: Real
    with out_buf.map_to_host() as h:
        speed = h[0]
    return speed


def max_wave_speed_euler[NDIM: Int, P: Int](
    var ctx: DeviceContext, mut q: DGState[NDIM, P, EULER_NC], app: Euler,
) raises -> Real:
    """max over nodes of `|u| + c`."""
    return _max_speed[NDIM, P, EULER_NC, SPEED_EULER](
        ctx, q,
        kernel_params(app.gas_gamma, app.min_density, app.min_pressure),
    )


def max_wave_speed_imhd[NDIM: Int, P: Int](
    var ctx: DeviceContext, mut q: DGState[NDIM, P, MHD_NC], app: IMHD,
) raises -> Real:
    """max over nodes of `|u| + sqrt(c_s² + c_A²)`."""
    return _max_speed[NDIM, P, MHD_NC, SPEED_IMHD](
        ctx, q,
        kernel_params(app.gas_gamma, app.min_density, app.min_pressure),
    )


def max_wave_speed_imhd_glm[NDIM: Int, P: Int](
    var ctx: DeviceContext, mut q: DGState[NDIM, P, MHD_GLM_NC], app: ImhdGLM,
) raises -> Real:
    """max over nodes of the ideal-MHD bound, floored at `c_h`."""
    return _max_speed[NDIM, P, MHD_GLM_NC, SPEED_IMHD_GLM](
        ctx, q,
        kernel_params(
            app.gas_gamma, app.min_density, app.min_pressure, app.c_h,
        ),
    )
