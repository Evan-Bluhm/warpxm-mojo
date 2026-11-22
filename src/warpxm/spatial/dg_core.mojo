# Nodal DG spatial solver core, generic over the dimension and
# parameterised on per-physics flux and BC closures (the "AppType"
# abstraction from upstream WARPXM).
#
# Pattern:
#   1.  Each physics declares one GPU-launchable kernel
#       `_<phys>_rhs_kernel[NDIM, P]` (in `dg_<phys>.mojo`) with the fixed
#       signature `launch_dg` expects.
#   2.  Inside it the kernel defines five closures (`def ... {imm}`):
#       `flux_x`, `flux_y`, `flux_z`, `num_flux`, `apply_bc`. They capture
#       the physics scalars and call the app's static flux methods.
#   3.  The kernel calls `_dg_compute_rhs_core[NDIM, P, NC, ...]`, which
#       does all the DG work. Only the first NDIM flux closures are ever
#       called; the rest are dead code.
#
# Every loop over axes is a `comptime for`, each closure has its own static
# type and the core is `@always_inline`, so each (physics, NDIM, P)
# instantiation compiles to straight-line code with nothing left of the
# genericity.
#
# ## Thread layout and element packing
#
# One thread per node. `ELEMS_PER_BLOCK` (EB) independent elements share a
# block, so that small elements still fill a warp:
#
#   1D  block_dim = (P1, EB)            thread = (in, eb)
#   2D  block_dim = (P1, P1, EB)        thread = (in, jn, eb)
#   3D  block_dim = (P1, P1, P1 · EB)   thread = (in, jn, eb · P1 + kn)
#
# Element index: `e = block_idx.x · EB + eb`; the grid has
# `ceildiv(num_elements, EB)` blocks. Shared memory holds the node fluxes
# of all EB elements and stays well under the per-block limit for every
# (NC, P) combination we ship.
#
# ## Floating-point evaluation order
#
# In 3D the volume and surface terms accumulate into one array (fewer live
# doubles per thread: with NC = 9 this saves 18 registers). In 1D and 2D
# each axis keeps its own derivative array and the terms are combined at
# the end. The two orders differ at round-off, and tests of near-vacuum
# problems (`test_euler_123_2d`) are sensitive to that, so each dimension
# keeps the order it has always had.

from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block import GHOST_NEG, GHOST_POS


comptime DG_CORE_ELEMS_PER_BLOCK: Int = 4
"""Elements packed into one block. Chosen on the RTX 3090 baseline (see
`bench/BASELINE.md`)."""

comptime InvJacobian = SIMD[WXM_DTYPE, 4]
"""`1 / jacobian` per axis (lanes beyond NDIM unused)."""


@always_inline
def _load_node[
    NDIM: Int, P: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    local_node: Int,
) -> Array[Real, NC]:
    comptime assert q.flat_rank == 1
    comptime num_nodes_per_elem = (P + 1) ** NDIM
    var base = (elem * num_nodes_per_elem + local_node) * NC
    var out: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out^


@always_inline
def _index_along[axis: Int](in_: Int, jn: Int, kn: Int) -> Int:
    """This node's index along `axis`."""
    comptime match axis:
        case 0:
            return in_
        case 1:
            return jn
        case _:
            return kn


@always_inline
def _node_on_line[P: Int, axis: Int](in_: Int, jn: Int, kn: Int, m: Int) -> Int:
    """Local node number of the node at position `m` along `axis`, on the
    line through node `(in_, jn, kn)`."""
    comptime P1 = P + 1
    comptime match axis:
        case 0:
            return (kn * P1 + jn) * P1 + m
        case 1:
            return (kn * P1 + m) * P1 + in_
        case _:
            return (m * P1 + jn) * P1 + in_


@always_inline
def _dg_compute_rhs_core[
    NDIM: Int,
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
    inv_jacobian: InvJacobian,
    num_elements: Int,
    flux_x_fn: FluxX,
    flux_y_fn: FluxY,
    flux_z_fn: FluxZ,
    num_flux_fn: NumFlux,
    apply_bc_fn: ApplyBc,
):
    """Per-node DG residual for an NC-component conservation law in NDIM
    dimensions: volume term by tensor-product GLL differentiation, surface
    term through `num_flux_fn(q_neg, q_pos, nx, ny, nz)` on the unit normal
    of each axis. `apply_bc_fn(q_interior, bc_kind, face_axis)` supplies
    the ghost state on a boundary face.
    """
    comptime assert NDIM >= 1 and NDIM <= 3, "NDIM must be 1, 2 or 3"
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 ** NDIM
    comptime faces_per_elem = 2 * NDIM
    comptime EB = ELEMS_PER_BLOCK
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    # Node indices along each axis (0 beyond NDIM) and element-in-block.
    var in_ = Int(thread_idx.x)
    var jn = 0
    var kn = 0
    var eb_in_block: Int
    comptime match NDIM:
        case 1:
            eb_in_block = Int(thread_idx.y)
        case 2:
            jn = Int(thread_idx.y)
            eb_in_block = Int(thread_idx.z)
        case _:
            jn = Int(thread_idx.y)
            var tz = Int(thread_idx.z)
            eb_in_block = tz // P1
            kn = tz % P1
    var e = Int(block_idx.x) * EB + eb_in_block

    # Partial last block: threads past the last element must still reach
    # the barrier below, at the same call site as everyone else. A barrier
    # on a divergent path deadlocks the warp, so there is no early return
    # here: such a thread redoes the last element's flux into its own
    # shared-memory slot and stops after the barrier, before any write.
    # (The block is exactly P1 wide along each node axis, so the node
    # indices are always in range.)
    var past_end = e >= num_elements
    if past_end:
        e = num_elements - 1

    var local_n = (kn * P1 + jn) * P1 + in_

    # Shared-memory flux cache, one slot per element-in-block.
    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[EB, nodes_per_elem, NC, NDIM]())

    # Flux of this node along each axis, `f_n[axis · NC + c]`.
    var q_n = _load_node[NDIM, P, NC](q, e, local_n)
    var f_n = Array[Real, NDIM * NC](fill=Real(0.0))
    comptime for axis in range(NDIM):
        var f_axis: Array[Real, NC]
        comptime match axis:
            case 0:
                f_axis = flux_x_fn(q_n)
            case 1:
                f_axis = flux_y_fn(q_n)
            case _:
                f_axis = flux_z_fn(q_n)
        comptime for c in range(NC):
            f_n[axis * NC + c] = f_axis[c]
            f_shared[eb_in_block, local_n, c, axis] = rebind[
                f_shared.ElementType
            ](f_axis[c])

    barrier()
    if past_end:
        return

    # --- Volume term: ∇·f at this node by tensor-product differentiation ---
    # See "Floating-point evaluation order" in the header: 3D folds
    # everything into `accum`; 1D / 2D keep one derivative array per axis
    # (`deriv`) and use `accum` for the surface term only.
    comptime SINGLE_ACCUMULATOR = NDIM == 3
    var accum = Array[Real, NC](fill=Real(0.0))
    var deriv = Array[Real, NDIM * NC](fill=Real(0.0))

    comptime for axis in range(NDIM):
        var along = _index_along[axis](in_, jn, kn)
        comptime for m in range(P1):
            var d: Scalar[WXM_DTYPE] = NodalBasis1D[P].diff_entry[m](along)
            comptime if SINGLE_ACCUMULATOR:
                d = d * inv_jacobian[axis]
            var m_node = _node_on_line[P, axis](in_, jn, kn, m)
            comptime for c in range(NC):
                var f_m = rebind[Scalar[WXM_DTYPE]](
                    f_shared[eb_in_block, m_node, c, axis]
                )
                comptime if SINGLE_ACCUMULATOR:
                    accum[c] -= d * f_m
                else:
                    deriv[axis * NC + c] += d * f_m

    # --- Surface term: only nodes on an element face contribute ------------
    # The launcher passes 1 / jacobian and the weights enter as comptime
    # reciprocals, so the kernel only multiplies: a double division costs
    # roughly ten FP64 instructions on the device.
    comptime inv_w0: Real = Real(1.0) / NodalBasis1D[P].weights()[0]
    comptime inv_wP: Real = Real(1.0) / NodalBasis1D[P].weights()[P]

    comptime for axis in range(NDIM):
        var along = _index_along[axis](in_, jn, kn)
        comptime nx: Real = 1.0 if axis == 0 else 0.0
        comptime ny: Real = 1.0 if axis == 1 else 0.0
        comptime nz: Real = 1.0 if axis == 2 else 0.0
        var inv_jac: Scalar[WXM_DTYPE] = inv_jacobian[axis]

        if along == 0:
            var f_id = Int(rebind[Scalar[INDEX_DTYPE]](
                elem_face_dir[e * faces_per_elem + 2 * axis]
            ))
            var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
            var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
            var q_nbr: Array[Real, NC]
            if neg_e == Int(GHOST_NEG):
                q_nbr = apply_bc_fn(q_n, bc_kind, axis)
            else:
                q_nbr = _load_node[NDIM, P, NC](
                    q, neg_e, _node_on_line[P, axis](in_, jn, kn, P),
                )
            var f_star = num_flux_fn(q_nbr, q_n, nx, ny, nz)
            var scale: Scalar[WXM_DTYPE] = inv_jac * inv_w0
            comptime for c in range(NC):
                accum[c] += (f_star[c] - f_n[axis * NC + c]) * scale

        if along == P:
            var f_id = Int(rebind[Scalar[INDEX_DTYPE]](
                elem_face_dir[e * faces_per_elem + 2 * axis + 1]
            ))
            var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
            var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
            var q_nbr: Array[Real, NC]
            if pos_e == Int(GHOST_POS):
                q_nbr = apply_bc_fn(q_n, bc_kind, axis)
            else:
                q_nbr = _load_node[NDIM, P, NC](
                    q, pos_e, _node_on_line[P, axis](in_, jn, kn, 0),
                )
            var f_star = num_flux_fn(q_n, q_nbr, nx, ny, nz)
            var scale: Scalar[WXM_DTYPE] = inv_jac * inv_wP
            comptime for c in range(NC):
                accum[c] += (f_n[axis * NC + c] - f_star[c]) * scale

    # --- Write back ---------------------------------------------------------
    var base = (e * nodes_per_elem + local_n) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = accum[c]
        comptime if not SINGLE_ACCUMULATOR:
            v = -deriv[c] * inv_jacobian[0]
            comptime for axis in range(NDIM):
                comptime if axis >= 1:
                    v = v - deriv[axis * NC + c] * inv_jacobian[axis]
            v = v + accum[c]
        rhs[base + c] = rebind[rhs.ElementType](v)
