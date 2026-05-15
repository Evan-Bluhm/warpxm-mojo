# Unified 2D nodal DG spatial solver core, parameterised on per-physics
# flux and BC closures (the "AppType" abstraction from upstream WARPXM).
#
# Pattern:
#   1.  Each per-physics kernel file declares a GPU-launchable
#       `def _<phys>_rhs_kernel_2d[P, ...]` that takes the physics-
#       specific scalar parameters (γ, c_light, etc.) as kernel arguments.
#   2.  Inside that kernel it defines four `@parameter fn` closures with
#       uniform signatures: `flux_x`, `flux_y`, `num_flux`, `apply_bc`.
#       The closures capture the kernel's scalar args and call into the
#       app's static flux methods.
#   3.  The kernel then calls `_dg_2d_compute_rhs_core[..., flux_x,
#       flux_y, num_flux, apply_bc]` which does all the DG work.
#
# Every closure is `@parameter` and the core is `@always_inline`, so the
# generated GPU code is byte-identical to the hand-rolled per-physics
# kernels we replace.

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import (
    GHOST_NEG, GHOST_POS,
    FACE_DIR_LEFT, FACE_DIR_RIGHT, FACE_DIR_BOTTOM, FACE_DIR_TOP,
)


@always_inline
def _load_node_2d[
    P: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    local_node: Int,
) -> InlineArray[Real, NC]:
    comptime assert q.flat_rank == 1
    comptime num_nodes_per_elem = (P + 1) * (P + 1)
    var base = (elem * num_nodes_per_elem + local_node) * NC
    var out: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out


@always_inline
def _dg_2d_compute_rhs_core[
    P: Int, NC: Int,
    flux_x_fn: def(InlineArray[Real, NC]) capturing [_] -> InlineArray[Real, NC],
    flux_y_fn: def(InlineArray[Real, NC]) capturing [_] -> InlineArray[Real, NC],
    num_flux_fn: def(
        InlineArray[Real, NC], InlineArray[Real, NC], Real, Real,
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
    jacobian_x: Real,
    jacobian_y: Real,
    num_elements: Int,
):
    """Generic per-node DG residual for a 2D NC-component conservation
    law. Volume term via tensor-product GLL differentiation, surface
    term via the supplied numerical-flux closure.

    Thread layout: `block_idx.x = elem`, `(thread_idx.x, thread_idx.y) =
    (in_, jn)` where `(in_, jn)` are the node indices within the element.
    """
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    var e = Int(block_idx.x)
    var in_ = Int(thread_idx.x)
    var jn = Int(thread_idx.y)
    if e >= num_elements or in_ >= P1 or jn >= P1:
        return

    var local_n = jn * P1 + in_

    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[P1 * P1, NC, 2]())

    var q_n = _load_node_2d[P, NC](q, e, local_n)
    var fx_n = flux_x_fn(q_n)
    var fy_n = flux_y_fn(q_n)
    comptime for c in range(NC):
        f_shared[local_n, c, 0] = rebind[f_shared.ElementType](fx_n[c])
        f_shared[local_n, c, 1] = rebind[f_shared.ElementType](fy_n[c])

    barrier()

    # --- Volume term: ∇·f at this node from tensor-product differentiation --
    var d_flat = NodalBasis1D[P].diff_matrix_flat()
    var dfdx: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    var dgdy: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    comptime for kn in range(P1):
        var dx_in_kn: Scalar[WXM_DTYPE] = d_flat[in_ * P1 + kn]
        var k_node = jn * P1 + kn
        comptime for c in range(NC):
            var fx_kn = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 0])
            dfdx[c] += dx_in_kn * fx_kn

    comptime for ln in range(P1):
        var dy_jn_ln: Scalar[WXM_DTYPE] = d_flat[jn * P1 + ln]
        var l_node = ln * P1 + in_
        comptime for c in range(NC):
            var fy_ln = rebind[Scalar[WXM_DTYPE]](f_shared[l_node, c, 1])
            dgdy[c] += dy_jn_ln * fy_ln

    # --- Surface term: only the four face-boundary nodes contribute --------
    var weights = NodalBasis1D[P].weights()
    var w_0: Scalar[WXM_DTYPE] = weights[0]
    var w_P: Scalar[WXM_DTYPE] = weights[P]

    var surface: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    if in_ == 0:
        var fL = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_LEFT]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fL]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fL]))
        var q_neg: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG):
            q_neg = apply_bc_fn(q_n, bc_kind, 0)
        else:
            q_neg = _load_node_2d[P, NC](q, neg_e, jn * P1 + P)
        var f_star = num_flux_fn(q_neg, q_n, Real(1.0), Real(0.0))
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fx_n[c]) / (jacobian_x * w_0)

    if in_ == P:
        var fR = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_RIGHT]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fR]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fR]))
        var q_pos: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS):
            q_pos = apply_bc_fn(q_n, bc_kind, 0)
        else:
            q_pos = _load_node_2d[P, NC](q, pos_e, jn * P1 + 0)
        var f_star = num_flux_fn(q_n, q_pos, Real(1.0), Real(0.0))
        comptime for c in range(NC):
            surface[c] += (fx_n[c] - f_star[c]) / (jacobian_x * w_P)

    if jn == 0:
        var fB = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_BOTTOM]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fB]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fB]))
        var q_neg: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG):
            q_neg = apply_bc_fn(q_n, bc_kind, 1)
        else:
            q_neg = _load_node_2d[P, NC](q, neg_e, P * P1 + in_)
        var f_star = num_flux_fn(q_neg, q_n, Real(0.0), Real(1.0))
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fy_n[c]) / (jacobian_y * w_0)

    if jn == P:
        var fT = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_TOP]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fT]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fT]))
        var q_pos: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS):
            q_pos = apply_bc_fn(q_n, bc_kind, 1)
        else:
            q_pos = _load_node_2d[P, NC](q, pos_e, 0 * P1 + in_)
        var f_star = num_flux_fn(q_n, q_pos, Real(0.0), Real(1.0))
        comptime for c in range(NC):
            surface[c] += (fy_n[c] - f_star[c]) / (jacobian_y * w_P)

    # --- Write back ---------------------------------------------------------
    var base = (e * nodes_per_elem + local_n) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = (
            -dfdx[c] / jacobian_x - dgdy[c] / jacobian_y + surface[c]
        )
        rhs[base + c] = rebind[rhs.ElementType](v)
