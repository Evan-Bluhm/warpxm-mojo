# 1D nodal DG spatial solver for Maxwell — thin wrapper over the
# shared `_dg_1d_compute_rhs_core` template.
#
# A "1D Maxwell" problem propagates along the single mesh axis (x̂),
# with the four other field components (E_y, E_z, B_y, B_z) coupling
# in the x-flux. E_x and B_x are stationary in 1D since ∂/∂y = ∂/∂z = 0.

from std.collections import InlineArray
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT, BC_PEC_WALL, BC_PMC_WALL
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_core_1d import _dg_1d_compute_rhs_core
from warpxm.variable.state_1d import DGState1D


@always_inline
def _apply_bc_maxwell_1d(
    q_interior: InlineArray[Real, MAXWELL_NC], bc_kind: Int,
) -> InlineArray[Real, MAXWELL_NC]:
    """Ghost-state synthesis for 1D Maxwell.

    `BC_PEC_WALL` (perfect electric conductor): wall normal is ±x̂, so
    flip the two tangential E components (E_y, E_z) and the normal B
    (B_x). `BC_PMC_WALL` (perfect magnetic conductor): flip normal E
    (E_x) and the two tangential B (B_y, B_z). Anything else copies.
    """
    var q: InlineArray[Real, MAXWELL_NC] = q_interior
    if bc_kind == BC_PEC_WALL:
        q[IDX_EY] = -q_interior[IDX_EY]
        q[IDX_EZ] = -q_interior[IDX_EZ]
        q[IDX_BX] = -q_interior[IDX_BX]
    elif bc_kind == BC_PMC_WALL:
        q[IDX_EX] = -q_interior[IDX_EX]
        q[IDX_BY] = -q_interior[IDX_BY]
        q[IDX_BZ] = -q_interior[IDX_BZ]
    return q


def _maxwell_rhs_kernel_1d[
    P: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    ElemFaceLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    elem_left_face: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    elem_right_face: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_left: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_right: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, ElemFaceLayout, MutAnyOrigin],
    c_light: Real,
    jacobian: Real,
    num_elements: Int,
):
    @parameter
    def flux_x(qn: InlineArray[Real, MAXWELL_NC]) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_x_static(qn, c_light)

    @parameter
    def num_flux(
        q_l: InlineArray[Real, MAXWELL_NC],
        q_r: InlineArray[Real, MAXWELL_NC],
        nx: Real, ny: Real, nz: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.numerical_flux_static(q_l, q_r, nx, ny, nz, c_light)

    @parameter
    def apply_bc(
        q_interior: InlineArray[Real, MAXWELL_NC], bc_kind: Int,
    ) -> InlineArray[Real, MAXWELL_NC]:
        return _apply_bc_maxwell_1d(q_interior, bc_kind)

    _dg_1d_compute_rhs_core[
        P, MAXWELL_NC, flux_x, num_flux, apply_bc,
        QLayout, RhsLayout, ElemFaceLayout,
    ](
        q, rhs, elem_left_face, elem_right_face,
        face_left, face_right, face_bc,
        jacobian, num_elements,
    )


struct DGMaxwellSolver1D[P: Int]:
    """Host-side launcher for the 1D Maxwell DG kernel."""

    var app: Maxwell

    def __init__(out self, app: Maxwell):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, MAXWELL_NC],
        mut rhs: DGState1D[Self.P, MAXWELL_NC],
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

        comptime kernel = _maxwell_rhs_kernel_1d[
            Self.P,
            type_of(q_layout), type_of(q_layout), type_of(elem_face_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, elf_view, erf_view, fl_view, fr_view, fbc_view,
            self.app.c_light, mesh.jacobian, mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=num_nodes_per_elem,
        )
