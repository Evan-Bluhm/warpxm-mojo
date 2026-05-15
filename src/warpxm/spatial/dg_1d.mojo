# 1D nodal DG spatial solver (GPU-only).
#
# Computes the right-hand side of `du/dt = rhs(u)` for an `NC`-component 1D
# hyperbolic conservation law `u_t + f_x(u) = 0` on a `BlockMesh1D`, using a
# strong-form nodal DG discretization with GLL collocation.
#
# This is the analogue of `ndg_kokkos_t` in upstream WARPXM, except:
#
# * No app-registry indirection — the physics app is a compile-time template
#   parameter.
# * No multi-app batching — one solver instance per app, for now.
# * Mass matrix is the (diagonal) GLL lumped form.
#
# Strong-form semi-discrete update on element e, node i, component c:
#
#     du_{i,c}/dt = -(1/J) (D @ f_c)_i
#                   + (1/(J w_i)) · [ δ_{i,P}·(f_{P,c} - f*_{R,c})
#                                    + δ_{i,0}·(f*_{L,c} - f_{0,c}) ]
#
# where J = dx/2 (constant on uniform mesh), D is the GLL differentiation
# matrix on [-1, 1], w_i are the GLL quadrature weights, and f*_{L,c} /
# f*_{R,c} are the single-valued numerical fluxes at the element's left /
# right faces for component c.

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, Idx, row_major, stack_allocation

from warpxm.apps.advection import Advection1D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D, GHOST_LEFT, GHOST_RIGHT
from warpxm.variable.state_1d import DGState1D


@always_inline
def _apply_bc_scalar[NC: Int](
    q_interior: InlineArray[Real, NC], bc_kind: Int,
) -> InlineArray[Real, NC]:
    """Synthesize the ghost-side state from an interior state for a scalar
    advection boundary face.

    For scalar (non-momentum-carrying) advected fields, only `BC_COPYOUT` is
    physically meaningful — reflecting walls don't apply to passive scalars.
    Any other BC kind falls back to copyout here; richer per-app BC dispatch
    belongs in the physics-specific kernel (e.g. Euler).
    """
    if bc_kind == BC_COPYOUT:
        return q_interior
    # Default — also copyout. Free-slip / no-slip / Dirichlet for scalars are
    # all equivalent to "match the interior" for a passive advected field.
    return q_interior


# ---------------------------------------------------------------------------
# Helpers — pull the NC-component state at a given node into a stack-local
# InlineArray.
# ---------------------------------------------------------------------------

@always_inline
def _load_node[
    P: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    node: Int,
) -> InlineArray[Real, NC]:
    """Read all NC components of (elem, node) into a small register array."""
    comptime assert q.flat_rank == 1, "_load_node expects a 1D tensor"
    comptime num_nodes_per_elem = P + 1
    var base = (elem * num_nodes_per_elem + node) * NC
    var out: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out


# ---------------------------------------------------------------------------
# GPU kernel: compute du/dt for `Advection1D[NC]` on a periodic 1D mesh.
#
# Launch convention: grid_dim = num_elements, block_dim = num_nodes_per_elem.
# Each thread maps to one node within one element, and the block cooperates
# through shared memory to evaluate the internal flux + differentiation
# matrix product. Numerical fluxes at the element's two faces are computed
# redundantly on the boundary threads (i = 0 and i = P) — wasteful but
# straightforward; we can optimize later by hoisting face-flux to its own
# kernel.
# ---------------------------------------------------------------------------

def _advection_rhs_kernel[
    P: Int,
    NC: Int,
    QLayout: TensorLayout,
    RhsLayout: TensorLayout,
    ElemFaceLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    elem_left_face: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    elem_right_face: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_left: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_right: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    velocity: Real,
    jacobian: Real,
    num_elements: Int,
    num_faces: Int,
):
    comptime num_nodes_per_elem = P + 1
    comptime assert q.flat_rank == 1, "DG kernel expects 1D q tensor"
    comptime assert rhs.flat_rank == 1, "DG kernel expects 1D rhs tensor"
    comptime assert elem_left_face.flat_rank == 1
    comptime assert elem_right_face.flat_rank == 1
    comptime assert face_left.flat_rank == 1
    comptime assert face_right.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    var e = Int(block_idx.x)
    var i = Int(thread_idx.x)
    if e >= num_elements or i >= num_nodes_per_elem:
        return

    # Shared internal-flux scratch indexed [node, component], row-major.
    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[num_nodes_per_elem, NC]())

    # ------------------------------------------------------------------
    # 1) Internal flux at this thread's node, for every component.
    # ------------------------------------------------------------------
    var q_i = _load_node[P, NC](q, e, i)
    var f_i = Advection1D[NC].internal_flux(q_i, velocity)
    comptime for c in range(NC):
        f_shared[i, c] = rebind[f_shared.ElementType](f_i[c])

    barrier()

    # ------------------------------------------------------------------
    # 2) Volume term: (D @ f)_i for each component.
    # ------------------------------------------------------------------
    var d_flat = NodalBasis1D[P].diff_matrix_flat()
    var dfdxi: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    comptime for j in range(num_nodes_per_elem):
        var d_ij: Scalar[WXM_DTYPE] = d_flat[i * num_nodes_per_elem + j]
        comptime for c in range(NC):
            var fj_c = rebind[Scalar[WXM_DTYPE]](f_shared[j, c])
            dfdxi[c] += d_ij * fj_c

    # ------------------------------------------------------------------
    # 3) Surface term: face-flux corrections at i=0 (left face) and i=P
    #    (right face).
    # ------------------------------------------------------------------
    var weights = NodalBasis1D[P].weights()
    var surface: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    if i == 0:
        var fL = Int(rebind[Scalar[INDEX_DTYPE]](elem_left_face[e]))
        var le = Int(rebind[Scalar[INDEX_DTYPE]](face_left[fL]))
        var re = Int(rebind[Scalar[INDEX_DTYPE]](face_right[fL]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fL]))

        var q_neg: InlineArray[Real, NC]
        if le == Int(GHOST_LEFT):
            # Left domain boundary; synthesize ghost from the interior element
            # `re` (which is `e`, since this element owns the boundary face).
            var q_interior = _load_node[P, NC](q, re, 0)
            q_neg = _apply_bc_scalar[NC](q_interior, bc_kind)
        else:
            q_neg = _load_node[P, NC](q, le, num_nodes_per_elem - 1)

        var q_pos = _load_node[P, NC](q, re, 0)
        var f_star = Advection1D[NC].numerical_flux(q_neg, q_pos, Real(1.0), velocity)
        var f_int = Advection1D[NC].internal_flux(q_i, velocity)
        var w0: Scalar[WXM_DTYPE] = weights[0]
        comptime for c in range(NC):
            surface[c] = (f_star[c] - f_int[c]) / (jacobian * w0)

    if i == (num_nodes_per_elem - 1):
        var fR = Int(rebind[Scalar[INDEX_DTYPE]](elem_right_face[e]))
        var le = Int(rebind[Scalar[INDEX_DTYPE]](face_left[fR]))
        var re = Int(rebind[Scalar[INDEX_DTYPE]](face_right[fR]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fR]))

        var q_neg = _load_node[P, NC](q, le, num_nodes_per_elem - 1)
        var q_pos: InlineArray[Real, NC]
        if re == Int(GHOST_RIGHT):
            var q_interior = _load_node[P, NC](q, le, num_nodes_per_elem - 1)
            q_pos = _apply_bc_scalar[NC](q_interior, bc_kind)
        else:
            q_pos = _load_node[P, NC](q, re, 0)

        var f_star = Advection1D[NC].numerical_flux(q_neg, q_pos, Real(1.0), velocity)
        var f_int = Advection1D[NC].internal_flux(q_i, velocity)
        var wP: Scalar[WXM_DTYPE] = weights[num_nodes_per_elem - 1]
        comptime for c in range(NC):
            surface[c] = (f_int[c] - f_star[c]) / (jacobian * wP)

    # ------------------------------------------------------------------
    # 4) Assemble RHS, one component at a time.
    # ------------------------------------------------------------------
    var base = (e * num_nodes_per_elem + i) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = -dfdxi[c] / jacobian + surface[c]
        rhs[base + c] = rebind[rhs.ElementType](v)


# ---------------------------------------------------------------------------
# Host-side driver.
# ---------------------------------------------------------------------------

struct DGSpatialSolver1D[P: Int, NC: Int]:
    """Host-side launcher for the multi-component 1D advection DG kernel."""

    var app: Advection1D[Self.NC]

    def __init__(out self, app: Advection1D[Self.NC]):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, Self.NC],
        mut rhs: DGState1D[Self.P, Self.NC],
    ) raises:
        """Launch the RHS kernel; result enqueues into `rhs.buf`."""
        comptime num_nodes_per_elem = Self.P + 1

        var q_layout = row_major(Idx(q.buf_len))
        var face_layout = row_major(Idx(mesh.num_faces))
        var elem_face_layout = row_major(Idx(mesh.num_elements))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var elf_view = TileTensor(mesh.elem_left_face_buf, elem_face_layout)
        var erf_view = TileTensor(mesh.elem_right_face_buf, elem_face_layout)
        var fl_view = TileTensor(mesh.face_left_buf, face_layout)
        var fr_view = TileTensor(mesh.face_right_buf, face_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _advection_rhs_kernel[
            Self.P, Self.NC,
            type_of(q_layout),
            type_of(q_layout),
            type_of(elem_face_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, elf_view, erf_view, fl_view, fr_view, fbc_view,
            self.app.velocity_x, mesh.jacobian, mesh.num_elements, mesh.num_faces,
            grid_dim=mesh.num_elements,
            block_dim=num_nodes_per_elem,
        )
