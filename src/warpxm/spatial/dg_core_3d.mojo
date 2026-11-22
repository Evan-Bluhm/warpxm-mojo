# Unified 3D nodal DG spatial solver core. Mirror of `dg_core_2d.mojo`:
# a generic NC-parametric kernel template that takes per-physics flux
# and BC closures, used by `dg_euler_3d.mojo`, `dg_maxwell_3d.mojo`,
# `dg_imhd_3d.mojo`, `dg_imhd_glm_3d.mojo`, and the advection 3D solver.
#
# Six face corrections per element (one per cube face): LEFT/RIGHT,
# BOTTOM/TOP, DOWN/UP.
#
# ## Element packing (DG-perf Task 1)
#
# `ELEMS_PER_BLOCK` controls how many independent elements share a CUDA
# block. The native 3D block has `(P+1)³` threads — already 27 for P=2
# without any further axis to pack onto. We grow `block_dim.z` from
# `P1` to `P1 · ELEMS_PER_BLOCK` and split it as
# `thread_idx.z = eb_in_block · P1 + kn`. Shared memory grows by
# `ELEMS_PER_BLOCK`× but stays well under the 100 KB/SM limit.

from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import (
    GHOST_NEG_3D, GHOST_POS_3D,
    FACE_DIR_LEFT_3D, FACE_DIR_RIGHT_3D,
    FACE_DIR_BOTTOM_3D, FACE_DIR_TOP_3D,
    FACE_DIR_DOWN_3D, FACE_DIR_UP_3D,
)


# Default number of elements packed into one CUDA block for the 3D DG
# core. Per-physics wrappers may override. EB=2 lands `2 · 27 = 54`
# threads/block (≈ 1.7 warps) for P=2, vs the 27-thread baseline.
comptime DG_CORE_3D_ELEMS_PER_BLOCK: Int = 4


@always_inline
def _load_node_3d[
    P: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    local_node: Int,
) -> Array[Real, NC]:
    comptime assert q.flat_rank == 1
    comptime num_nodes_per_elem = (P + 1) ** 3
    var base = (elem * num_nodes_per_elem + local_node) * NC
    var out: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out^


@always_inline
def _dg_3d_compute_rhs_core[
    P: Int,
    NC: Int,
    QLayout: TensorLayout,
    RhsLayout: TensorLayout,
    FaceLayout: TensorLayout,
    EFLayout: TensorLayout,
    ELEMS_PER_BLOCK: Int,
    FluxX: def(Array[Real, NC]) -> Array[Real, NC],
    FluxY: def(Array[Real, NC]) -> Array[Real, NC],
    FluxZ: def(Array[Real, NC]) -> Array[Real, NC],
    NumFlux: def(
        Array[Real, NC], Array[Real, NC], Real, Real, Real,
    ) -> Array[Real, NC],
    ApplyBc: def(
        Array[Real, NC], Int, Int,
    ) -> Array[Real, NC],
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    inv_jacobian_x: Real, inv_jacobian_y: Real, inv_jacobian_z: Real,
    num_elements: Int,
    flux_x_fn: FluxX,
    flux_y_fn: FluxY,
    flux_z_fn: FluxZ,
    num_flux_fn: NumFlux,
    apply_bc_fn: ApplyBc,
):
    """Generic per-node DG residual for a 3D NC-component conservation
    law. Volume term via tensor-product GLL differentiation, surface
    term via the supplied numerical-flux closure on a 3-component normal.

    Thread layout: `block_idx.x = elem_block`, `(thread_idx.x,
    thread_idx.y, thread_idx.z) = (in_, jn, eb · P1 + kn)`.
    `eb_in_block = thread_idx.z / P1`, `kn = thread_idx.z % P1`.

    Element index: `e = block_idx.x · ELEMS_PER_BLOCK + eb_in_block`.
    Grid: `grid_dim = ceildiv(num_elements, ELEMS_PER_BLOCK)`.
    """
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1 * P1
    comptime EB = ELEMS_PER_BLOCK
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    var in_ = Int(thread_idx.x)
    var jn = Int(thread_idx.y)
    var tz = Int(thread_idx.z)
    var eb_in_block = tz // P1
    var kn = tz % P1
    var e = Int(block_idx.x) * EB + eb_in_block

    if e >= num_elements or in_ >= P1 or jn >= P1 or kn >= P1:
        barrier()
        return

    var local_n = kn * P1 * P1 + jn * P1 + in_

    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[EB, P1 * P1 * P1, NC, 3]())

    var q_n = _load_node_3d[P, NC](q, e, local_n)
    var fx_n = flux_x_fn(q_n)
    var fy_n = flux_y_fn(q_n)
    var fz_n = flux_z_fn(q_n)
    comptime for c in range(NC):
        f_shared[eb_in_block, local_n, c, 0] = rebind[f_shared.ElementType](fx_n[c])
        f_shared[eb_in_block, local_n, c, 1] = rebind[f_shared.ElementType](fy_n[c])
        f_shared[eb_in_block, local_n, c, 2] = rebind[f_shared.ElementType](fz_n[c])

    barrier()

    # --- Single-accumulator (DG-perf Task 2) -------------------------------
    # Volume terms for all three axes are folded directly into `accum`
    # immediately, so `dfdx`, `dgdy`, `dhdz` never exist simultaneously
    # as separate arrays. With NC=9, this saves 2·9 = 18 live doubles
    # per thread vs the 3-array form. Surface contributions are added
    # on top sequentially. The reorder breaks the FP bit-identity that
    # `test_imhd_briowu_2d_hlld`-style y-drift assertions rely on, so
    # any 3D bit-identity test (3D Alfvén, 3D Brio-Wu) needs to allow
    # for ~ULP-level drift.
    var inv_jac_x: Scalar[WXM_DTYPE] = inv_jacobian_x
    var inv_jac_y: Scalar[WXM_DTYPE] = inv_jacobian_y
    var inv_jac_z: Scalar[WXM_DTYPE] = inv_jacobian_z
    var accum: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))

    comptime for kx in range(P1):
        var d_in_kx: Scalar[WXM_DTYPE] = NodalBasis1D[P].diff_entry[kx](in_) * inv_jac_x
        var k_node = kn * P1 * P1 + jn * P1 + kx
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[eb_in_block, k_node, c, 0])
            accum[c] -= d_in_kx * v
    comptime for ky in range(P1):
        var d_jn_ky: Scalar[WXM_DTYPE] = NodalBasis1D[P].diff_entry[ky](jn) * inv_jac_y
        var k_node = kn * P1 * P1 + ky * P1 + in_
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[eb_in_block, k_node, c, 1])
            accum[c] -= d_jn_ky * v
    comptime for kz in range(P1):
        var d_kn_kz: Scalar[WXM_DTYPE] = NodalBasis1D[P].diff_entry[kz](kn) * inv_jac_z
        var k_node = kz * P1 * P1 + jn * P1 + in_
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[eb_in_block, k_node, c, 2])
            accum[c] -= d_kn_kz * v

    # Face scales as products of reciprocals: `inv_jac / w` is a double
    # division (about ten FP64 instructions) on every face node.
    comptime inv_w0: Real = Real(1.0) / NodalBasis1D[P].weights()[0]
    comptime inv_wP: Real = Real(1.0) / NodalBasis1D[P].weights()[P]

    if in_ == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_LEFT_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: Array[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 0)
        else:
            q_nbr = _load_node_3d[P, NC](q, neg_e, kn * P1 * P1 + jn * P1 + P)
        var f_star = num_flux_fn(q_nbr, q_n, Real(1.0), Real(0.0), Real(0.0))
        comptime for c in range(NC):
            accum[c] += (f_star[c] - fx_n[c]) * (inv_jac_x * inv_w0)
    if in_ == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_RIGHT_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: Array[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 0)
        else:
            q_nbr = _load_node_3d[P, NC](q, pos_e, kn * P1 * P1 + jn * P1 + 0)
        var f_star = num_flux_fn(q_n, q_nbr, Real(1.0), Real(0.0), Real(0.0))
        comptime for c in range(NC):
            accum[c] += (fx_n[c] - f_star[c]) * (inv_jac_x * inv_wP)
    if jn == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_BOTTOM_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: Array[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 1)
        else:
            q_nbr = _load_node_3d[P, NC](q, neg_e, kn * P1 * P1 + P * P1 + in_)
        var f_star = num_flux_fn(q_nbr, q_n, Real(0.0), Real(1.0), Real(0.0))
        comptime for c in range(NC):
            accum[c] += (f_star[c] - fy_n[c]) * (inv_jac_y * inv_w0)
    if jn == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_TOP_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: Array[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 1)
        else:
            q_nbr = _load_node_3d[P, NC](q, pos_e, kn * P1 * P1 + 0 * P1 + in_)
        var f_star = num_flux_fn(q_n, q_nbr, Real(0.0), Real(1.0), Real(0.0))
        comptime for c in range(NC):
            accum[c] += (fy_n[c] - f_star[c]) * (inv_jac_y * inv_wP)
    if kn == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_DOWN_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: Array[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 2)
        else:
            q_nbr = _load_node_3d[P, NC](q, neg_e, P * P1 * P1 + jn * P1 + in_)
        var f_star = num_flux_fn(q_nbr, q_n, Real(0.0), Real(0.0), Real(1.0))
        comptime for c in range(NC):
            accum[c] += (f_star[c] - fz_n[c]) * (inv_jac_z * inv_w0)
    if kn == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_UP_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: Array[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 2)
        else:
            q_nbr = _load_node_3d[P, NC](q, pos_e, 0 * P1 * P1 + jn * P1 + in_)
        var f_star = num_flux_fn(q_n, q_nbr, Real(0.0), Real(0.0), Real(1.0))
        comptime for c in range(NC):
            accum[c] += (fz_n[c] - f_star[c]) * (inv_jac_z * inv_wP)

    var base = (e * nodes_per_elem + local_n) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = accum[c]
        rhs[base + c] = rebind[rhs.ElementType](v)
