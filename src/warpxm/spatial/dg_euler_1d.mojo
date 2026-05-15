# 1D nodal DG spatial solver for 5-moment Euler.
#
# Near-clone of `dg_1d.mojo` (advection) — same per-element thread layout,
# same strong-form residual structure. `NC` is fixed at 5 (the conservative
# Euler component count) and the flux helpers come from `apps/euler_1d.mojo`,
# which depend on (γ, ρ_min, P_min).
#
# BC support: `BC_COPYOUT` and `BC_WALL_FREESLIP` are honoured at the
# left/right boundary faces; in 1D a free-slip wall reflects the x-momentum
# (`IDX_MX`) and copies everything else.

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, Idx, row_major, stack_allocation

from warpxm.apps.euler_1d import Euler1D
from warpxm.apps.euler_2d import EULER_NC, IDX_MX
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D, GHOST_LEFT, GHOST_RIGHT
from warpxm.variable.state_1d import DGState1D


@always_inline
def _load_node[
    P: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    node: Int,
) -> InlineArray[Real, EULER_NC]:
    comptime assert q.flat_rank == 1
    comptime num_nodes_per_elem = P + 1
    var base = (elem * num_nodes_per_elem + node) * EULER_NC
    var out: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
    comptime for c in range(EULER_NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out


@always_inline
def _apply_bc_euler_1d(
    q_interior: InlineArray[Real, EULER_NC], bc_kind: Int,
) -> InlineArray[Real, EULER_NC]:
    """Ghost state synthesis for a 1D Euler boundary face.

    Free-slip wall flips the x-momentum; everything else copies. Other BC
    kinds fall back to copyout.
    """
    var q: InlineArray[Real, EULER_NC] = q_interior
    if bc_kind == BC_WALL_FREESLIP:
        q[IDX_MX] = -q_interior[IDX_MX]
    return q


def _euler_rhs_kernel_1d[
    P: Int,
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
    gamma: Real,
    rho_min: Real,
    press_min: Real,
    jacobian: Real,
    num_elements: Int,
):
    comptime num_nodes_per_elem = P + 1
    comptime NC = EULER_NC
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert elem_left_face.flat_rank == 1
    comptime assert elem_right_face.flat_rank == 1
    comptime assert face_left.flat_rank == 1
    comptime assert face_right.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    var e = Int(block_idx.x)
    var i = Int(thread_idx.x)
    if e >= num_elements or i >= num_nodes_per_elem:
        return

    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[num_nodes_per_elem, NC]())

    var q_i = _load_node[P](q, e, i)
    var f_i = Euler1D.internal_flux_static(q_i, gamma, rho_min, press_min)
    comptime for c in range(NC):
        f_shared[i, c] = rebind[f_shared.ElementType](f_i[c])

    barrier()

    var d_flat = NodalBasis1D[P].diff_matrix_flat()
    var dfdxi: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    comptime for j in range(num_nodes_per_elem):
        var d_ij: Scalar[WXM_DTYPE] = d_flat[i * num_nodes_per_elem + j]
        comptime for c in range(NC):
            var fj_c = rebind[Scalar[WXM_DTYPE]](f_shared[j, c])
            dfdxi[c] += d_ij * fj_c

    var weights = NodalBasis1D[P].weights()
    var surface: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    if i == 0:
        var fL = Int(rebind[Scalar[INDEX_DTYPE]](elem_left_face[e]))
        var le = Int(rebind[Scalar[INDEX_DTYPE]](face_left[fL]))
        var re = Int(rebind[Scalar[INDEX_DTYPE]](face_right[fL]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fL]))

        var q_neg: InlineArray[Real, NC]
        if le == Int(GHOST_LEFT):
            q_neg = _apply_bc_euler_1d(q_i, bc_kind)
        else:
            q_neg = _load_node[P](q, le, num_nodes_per_elem - 1)
        var q_pos = q_i  # element e's leftmost node IS i = 0
        var f_star = Euler1D.numerical_flux_static(
            q_neg, q_pos, Real(1.0), Real(0.0), Real(0.0),
            gamma, rho_min, press_min,
        )
        var w0: Scalar[WXM_DTYPE] = weights[0]
        comptime for c in range(NC):
            surface[c] = (f_star[c] - f_i[c]) / (jacobian * w0)

    if i == (num_nodes_per_elem - 1):
        var fR = Int(rebind[Scalar[INDEX_DTYPE]](elem_right_face[e]))
        var le = Int(rebind[Scalar[INDEX_DTYPE]](face_left[fR]))
        var re = Int(rebind[Scalar[INDEX_DTYPE]](face_right[fR]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fR]))

        var q_neg = q_i  # element e's rightmost node IS i = P
        var q_pos: InlineArray[Real, NC]
        if re == Int(GHOST_RIGHT):
            q_pos = _apply_bc_euler_1d(q_i, bc_kind)
        else:
            q_pos = _load_node[P](q, re, 0)
        var f_star = Euler1D.numerical_flux_static(
            q_neg, q_pos, Real(1.0), Real(0.0), Real(0.0),
            gamma, rho_min, press_min,
        )
        var wP: Scalar[WXM_DTYPE] = weights[num_nodes_per_elem - 1]
        comptime for c in range(NC):
            surface[c] = (f_i[c] - f_star[c]) / (jacobian * wP)

    var base = (e * num_nodes_per_elem + i) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = -dfdxi[c] / jacobian + surface[c]
        rhs[base + c] = rebind[rhs.ElementType](v)


struct DGEulerSolver1D[P: Int]:
    """Host-side launcher for the 1D 5-moment Euler DG kernel."""

    var app: Euler1D

    def __init__(out self, app: Euler1D):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, EULER_NC],
        mut rhs: DGState1D[Self.P, EULER_NC],
    ) raises:
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

        comptime kernel = _euler_rhs_kernel_1d[
            Self.P,
            type_of(q_layout), type_of(q_layout),
            type_of(elem_face_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, elf_view, erf_view, fl_view, fr_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            mesh.jacobian, mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=num_nodes_per_elem,
        )
