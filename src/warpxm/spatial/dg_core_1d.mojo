# Unified 1D nodal DG spatial solver core. Mirror of `dg_core_2d.mojo`
# and `dg_core_3d.mojo`: a generic NC-parametric kernel template taking
# per-physics flux and BC closures.
#
# Two face corrections per element (left + right). The closure
# signatures match the higher-dim versions: numerical flux takes a
# 3-component normal `(nx, ny, nz)` with `(ny, nz) = (0, 0)` for 1D
# callers — the 0 lanes fold away.
#
# ## Element packing (DG-perf Task 1)
#
# `ELEMS_PER_BLOCK` lives on `thread_idx.y`. The unpacked layout has
# `(P+1) = 3` threads/block for P=2 — well under a warp; the worst
# starting point of all three dimensions. With EB=4, we get 12
# threads/block. EB=8 / 16 trigger hangs / divergence stalls on
# certain 1D Maxwell-coupled tests (e.g. `test_vlasov_maxwell_
# uniform_oscillator_1d1v`); the issue is Mojo-codegen-flaky rather
# than fundamental — EB=4 is the conservative ceiling for now.

from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import GHOST_LEFT, GHOST_RIGHT


# Default number of elements packed into one CUDA block for the 1D DG
# core. Per-physics wrappers may override.
comptime DG_CORE_1D_ELEMS_PER_BLOCK: Int = 4


@always_inline
def _load_node_1d[
    P: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    node: Int,
) -> Array[Real, NC]:
    comptime assert q.flat_rank == 1
    comptime num_nodes_per_elem = P + 1
    var base = (elem * num_nodes_per_elem + node) * NC
    var out: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out^


@always_inline
def _dg_1d_compute_rhs_core[
    P: Int, NC: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    ElemFaceLayout: TensorLayout,
    ELEMS_PER_BLOCK: Int,
    FluxX: def(Array[Real, NC]) -> Array[Real, NC],
    NumFlux: def(
        Array[Real, NC], Array[Real, NC], Real, Real, Real,
    ) -> Array[Real, NC],
    ApplyBC: def(Array[Real, NC], Int) -> Array[Real, NC],
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    elem_left_face: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    elem_right_face: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_left: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_right: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    jacobian: Real,
    num_elements: Int,
    flux_x_fn: FluxX,
    num_flux_fn: NumFlux,
    apply_bc_fn: ApplyBC,
):
    """Generic per-node DG residual for a 1D NC-component conservation
    law. Volume term via GLL differentiation; surface term via the
    supplied numerical-flux closure on a 3-component unit normal (1D
    callers always pass `(±1, 0, 0)`).

    Thread layout: `block_idx.x = elem_block`, `thread_idx.x = i`,
    `thread_idx.y = eb_in_block`. Element index
    `e = block_idx.x · ELEMS_PER_BLOCK + thread_idx.y`.
    """
    comptime num_nodes_per_elem = P + 1
    comptime EB = ELEMS_PER_BLOCK
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert elem_left_face.flat_rank == 1
    comptime assert elem_right_face.flat_rank == 1
    comptime assert face_left.flat_rank == 1
    comptime assert face_right.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    var eb_in_block = Int(thread_idx.y)
    var e = Int(block_idx.x) * EB + eb_in_block
    var i = Int(thread_idx.x)
    if e >= num_elements or i >= num_nodes_per_elem:
        barrier()
        return

    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[EB, num_nodes_per_elem, NC]())

    var q_i = _load_node_1d[P, NC](q, e, i)
    var f_i = flux_x_fn(q_i)
    comptime for c in range(NC):
        f_shared[eb_in_block, i, c] = rebind[f_shared.ElementType](f_i[c])

    barrier()

    # --- Volume term: (D @ f)_i ----------------------------------------
    var d_flat = NodalBasis1D[P].diff_matrix_flat()
    var dfdxi: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))
    comptime for j in range(num_nodes_per_elem):
        var d_ij: Scalar[WXM_DTYPE] = d_flat[i * num_nodes_per_elem + j]
        comptime for c in range(NC):
            var fj_c = rebind[Scalar[WXM_DTYPE]](f_shared[eb_in_block, j, c])
            dfdxi[c] += d_ij * fj_c

    # --- Surface term: left and right faces ----------------------------
    var weights = NodalBasis1D[P].weights()
    var surface: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))

    if i == 0:
        var fL = Int(rebind[Scalar[INDEX_DTYPE]](elem_left_face[e]))
        var le = Int(rebind[Scalar[INDEX_DTYPE]](face_left[fL]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fL]))
        var q_neg: Array[Real, NC]
        if le == Int(GHOST_LEFT):
            q_neg = apply_bc_fn(q_i, bc_kind)
        else:
            q_neg = _load_node_1d[P, NC](q, le, num_nodes_per_elem - 1)
        var f_star = num_flux_fn(q_neg, q_i, Real(1.0), Real(0.0), Real(0.0))
        var w0: Scalar[WXM_DTYPE] = weights[0]
        comptime for c in range(NC):
            surface[c] = (f_star[c] - f_i[c]) / (jacobian * w0)

    if i == (num_nodes_per_elem - 1):
        var fR = Int(rebind[Scalar[INDEX_DTYPE]](elem_right_face[e]))
        var re = Int(rebind[Scalar[INDEX_DTYPE]](face_right[fR]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fR]))
        var q_pos: Array[Real, NC]
        if re == Int(GHOST_RIGHT):
            q_pos = apply_bc_fn(q_i, bc_kind)
        else:
            q_pos = _load_node_1d[P, NC](q, re, 0)
        var f_star = num_flux_fn(q_i, q_pos, Real(1.0), Real(0.0), Real(0.0))
        var wP: Scalar[WXM_DTYPE] = weights[num_nodes_per_elem - 1]
        comptime for c in range(NC):
            surface[c] = (f_i[c] - f_star[c]) / (jacobian * wP)

    var base = (e * num_nodes_per_elem + i) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = -dfdxi[c] / jacobian + surface[c]
        rhs[base + c] = rebind[rhs.ElementType](v)
