# Unified 3D nodal DG spatial solver core. Mirror of `dg_core_2d.mojo`:
# a generic NC-parametric kernel template that takes per-physics flux
# and BC closures, used by `dg_euler_3d.mojo`, `dg_maxwell_3d.mojo`,
# `dg_imhd_3d.mojo`, `dg_imhd_glm_3d.mojo`, and the advection 3D solver.
#
# Six face corrections per element (one per cube face): LEFT/RIGHT,
# BOTTOM/TOP, DOWN/UP.

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import (
    GHOST_NEG_3D, GHOST_POS_3D,
    FACE_DIR_LEFT_3D, FACE_DIR_RIGHT_3D,
    FACE_DIR_BOTTOM_3D, FACE_DIR_TOP_3D,
    FACE_DIR_DOWN_3D, FACE_DIR_UP_3D,
)


@always_inline
def _load_node_3d[
    P: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    local_node: Int,
) -> InlineArray[Real, NC]:
    comptime assert q.flat_rank == 1
    comptime num_nodes_per_elem = (P + 1) ** 3
    var base = (elem * num_nodes_per_elem + local_node) * NC
    var out: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out


@always_inline
def _dg_3d_compute_rhs_core[
    P: Int, NC: Int,
    flux_x_fn: def(InlineArray[Real, NC]) capturing [_] -> InlineArray[Real, NC],
    flux_y_fn: def(InlineArray[Real, NC]) capturing [_] -> InlineArray[Real, NC],
    flux_z_fn: def(InlineArray[Real, NC]) capturing [_] -> InlineArray[Real, NC],
    num_flux_fn: def(
        InlineArray[Real, NC], InlineArray[Real, NC], Real, Real, Real,
    ) capturing [_] -> InlineArray[Real, NC],
    apply_bc_fn: def(
        InlineArray[Real, NC], Int, Int,
    ) capturing [_] -> InlineArray[Real, NC],
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    jacobian_x: Real, jacobian_y: Real, jacobian_z: Real,
    num_elements: Int,
):
    """Generic per-node DG residual for a 3D NC-component conservation
    law. Volume term via tensor-product GLL differentiation, surface
    term via the supplied numerical-flux closure on a 3-component normal.

    Thread layout: `block_idx.x = elem`, `(thread_idx.x, thread_idx.y,
    thread_idx.z) = (in_, jn, kn)`.
    """
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1 * P1
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    var e = Int(block_idx.x)
    var in_ = Int(thread_idx.x)
    var jn = Int(thread_idx.y)
    var kn = Int(thread_idx.z)
    if e >= num_elements or in_ >= P1 or jn >= P1 or kn >= P1:
        return

    var local_n = kn * P1 * P1 + jn * P1 + in_

    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[P1 * P1 * P1, NC, 3]())

    var q_n = _load_node_3d[P, NC](q, e, local_n)
    var fx_n = flux_x_fn(q_n)
    var fy_n = flux_y_fn(q_n)
    var fz_n = flux_z_fn(q_n)
    comptime for c in range(NC):
        f_shared[local_n, c, 0] = rebind[f_shared.ElementType](fx_n[c])
        f_shared[local_n, c, 1] = rebind[f_shared.ElementType](fy_n[c])
        f_shared[local_n, c, 2] = rebind[f_shared.ElementType](fz_n[c])

    barrier()

    var d_flat = NodalBasis1D[P].diff_matrix_flat()
    var dfdx: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    var dgdy: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    var dhdz: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    comptime for kx in range(P1):
        var d_in_kx: Scalar[WXM_DTYPE] = d_flat[in_ * P1 + kx]
        var k_node = kn * P1 * P1 + jn * P1 + kx
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 0])
            dfdx[c] += d_in_kx * v
    comptime for ky in range(P1):
        var d_jn_ky: Scalar[WXM_DTYPE] = d_flat[jn * P1 + ky]
        var k_node = kn * P1 * P1 + ky * P1 + in_
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 1])
            dgdy[c] += d_jn_ky * v
    comptime for kz in range(P1):
        var d_kn_kz: Scalar[WXM_DTYPE] = d_flat[kn * P1 + kz]
        var k_node = kz * P1 * P1 + jn * P1 + in_
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 2])
            dhdz[c] += d_kn_kz * v

    var weights = NodalBasis1D[P].weights()
    var w_0: Scalar[WXM_DTYPE] = weights[0]
    var w_P: Scalar[WXM_DTYPE] = weights[P]
    var surface: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    if in_ == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_LEFT_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 0)
        else:
            q_nbr = _load_node_3d[P, NC](q, neg_e, kn * P1 * P1 + jn * P1 + P)
        var f_star = num_flux_fn(q_nbr, q_n, Real(1.0), Real(0.0), Real(0.0))
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fx_n[c]) / (jacobian_x * w_0)
    if in_ == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_RIGHT_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 0)
        else:
            q_nbr = _load_node_3d[P, NC](q, pos_e, kn * P1 * P1 + jn * P1 + 0)
        var f_star = num_flux_fn(q_n, q_nbr, Real(1.0), Real(0.0), Real(0.0))
        comptime for c in range(NC):
            surface[c] += (fx_n[c] - f_star[c]) / (jacobian_x * w_P)
    if jn == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_BOTTOM_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 1)
        else:
            q_nbr = _load_node_3d[P, NC](q, neg_e, kn * P1 * P1 + P * P1 + in_)
        var f_star = num_flux_fn(q_nbr, q_n, Real(0.0), Real(1.0), Real(0.0))
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fy_n[c]) / (jacobian_y * w_0)
    if jn == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_TOP_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 1)
        else:
            q_nbr = _load_node_3d[P, NC](q, pos_e, kn * P1 * P1 + 0 * P1 + in_)
        var f_star = num_flux_fn(q_n, q_nbr, Real(0.0), Real(1.0), Real(0.0))
        comptime for c in range(NC):
            surface[c] += (fy_n[c] - f_star[c]) / (jacobian_y * w_P)
    if kn == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_DOWN_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 2)
        else:
            q_nbr = _load_node_3d[P, NC](q, neg_e, P * P1 * P1 + jn * P1 + in_)
        var f_star = num_flux_fn(q_nbr, q_n, Real(0.0), Real(0.0), Real(1.0))
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fz_n[c]) / (jacobian_z * w_0)
    if kn == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_UP_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = apply_bc_fn(q_n, bc_kind, 2)
        else:
            q_nbr = _load_node_3d[P, NC](q, pos_e, 0 * P1 * P1 + jn * P1 + in_)
        var f_star = num_flux_fn(q_n, q_nbr, Real(0.0), Real(0.0), Real(1.0))
        comptime for c in range(NC):
            surface[c] += (fz_n[c] - f_star[c]) / (jacobian_z * w_P)

    var base = (e * nodes_per_elem + local_n) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = (
            -dfdx[c] / jacobian_x
            - dgdy[c] / jacobian_y
            - dhdz[c] / jacobian_z
            + surface[c]
        )
        rhs[base + c] = rebind[rhs.ElementType](v)
