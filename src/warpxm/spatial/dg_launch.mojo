# Shared host-side launch plumbing for the per-physics DG kernels.
#
# Every `dg_<phys>.mojo` wrapper has the same job on the host: wrap the
# state and mesh connectivity buffers in `TileTensor` views, pick the
# launch grid, and enqueue its kernel. That is done once here, so a wrapper
# is left with what actually differs between physics: the flux closures
# and the boundary-condition function.
#
# To make one launcher serve every physics and dimension, all DG kernels
# share one signature:
#
#     (q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
#      inv_jacobian, num_elements, params)
#
#   * buffers are `RealTensor` / `IndexTensor` (a flat, runtime-sized
#     layout is the same type for every buffer length);
#   * `inv_jacobian` is an `InvJacobian` vector, one lane per axis;
#   * physics scalars travel in one `KernelParams` vector, which the kernel
#     unpacks by position (`var gamma = params[0]`). Build it on the host
#     with `kernel_params(...)`, in the same order.
#
# A wrapper's `compute_rhs` is then one call:
#
#     launch_dg[_euler_rhs_kernel[Self.NDIM, Self.P]](
#         ctx, mesh, q, rhs,
#         kernel_params(app.gas_gamma, app.min_density, app.min_pressure),
#     )

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.math import ceildiv

from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block import BlockMesh
from warpxm.spatial.dg_core import DG_CORE_ELEMS_PER_BLOCK, InvJacobian
from warpxm.variable.state import DGState


comptime FlatLayout = type_of(row_major(Int(0)))
"""Layout of every flat device buffer: rank 1, runtime length."""

comptime RealTensor = TileTensor[WXM_DTYPE, FlatLayout, MutAnyOrigin]
comptime IndexTensor = TileTensor[INDEX_DTYPE, FlatLayout, MutAnyOrigin]

comptime MAX_KERNEL_PARAMS: Int = 8
comptime KernelParams = SIMD[WXM_DTYPE, MAX_KERNEL_PARAMS]
"""Physics scalars for one kernel launch, by position."""


def kernel_params(*values: Real) -> KernelParams:
    """Pack up to `MAX_KERNEL_PARAMS` physics scalars for a DG kernel."""
    debug_assert(len(values) <= MAX_KERNEL_PARAMS, "too many kernel parameters")
    var params = KernelParams(0.0)
    for i in range(len(values)):
        params[i] = values[i]
    return params


def launch_dg[
    NDIM: Int, P: Int, NC: Int, //,
    kernel: def(
        RealTensor, RealTensor,
        IndexTensor, IndexTensor, IndexTensor, IndexTensor,
        InvJacobian, Int64, KernelParams,
    ) thin -> None,
](
    var ctx: DeviceContext,
    mesh: BlockMesh[NDIM, P],
    mut q: DGState[NDIM, P, NC],
    mut rhs: DGState[NDIM, P, NC],
    params: KernelParams,
) raises:
    """Enqueue a DG kernel on `mesh`: one thread per node, with
    `DG_CORE_ELEMS_PER_BLOCK` elements per block (thread layout in
    `dg_core.mojo`)."""
    comptime EB = DG_CORE_ELEMS_PER_BLOCK
    comptime P1 = P + 1
    comptime BLOCK_Y = EB if NDIM == 1 else P1
    comptime BLOCK_Z = 1 if NDIM == 1 else (EB if NDIM == 2 else P1 * EB)
    var q_layout = row_major(Int(q.buf_len))
    var face_layout = row_major(Int(mesh.num_faces))
    var ef_layout = row_major(Int(mesh.num_elements * 2 * NDIM))
    var inv_jacobian = InvJacobian(
        Real(1.0) / mesh.jacobian_x, Real(1.0) / mesh.jacobian_y,
        Real(1.0) / mesh.jacobian_z, Real(0.0),
    )
    ctx.enqueue_function[kernel](
        TileTensor(q.buf, q_layout),
        TileTensor(rhs.buf, q_layout),
        TileTensor(mesh.face_neg_elem_buf, face_layout),
        TileTensor(mesh.face_pos_elem_buf, face_layout),
        TileTensor(mesh.elem_face_dir_buf, ef_layout),
        TileTensor(mesh.face_bc_buf, face_layout),
        inv_jacobian, Int64(mesh.num_elements), params,
        grid_dim=ceildiv(mesh.num_elements, EB),
        block_dim=(P1, BLOCK_Y, BLOCK_Z),
    )
