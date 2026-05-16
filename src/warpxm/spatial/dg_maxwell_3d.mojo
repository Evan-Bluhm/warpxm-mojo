# 3D nodal DG spatial solver for Maxwell — thin wrapper over the shared
# `_dg_3d_compute_rhs_core` template.

from std.collections import InlineArray
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT, BC_PEC_WALL, BC_PMC_WALL
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_core_3d import _dg_3d_compute_rhs_core
from warpxm.variable.state_3d import DGState3D


@always_inline
def _apply_bc_maxwell_3d(
    q_interior: InlineArray[Real, MAXWELL_NC],
    bc_kind: Int,
    face_axis: Int,
) -> InlineArray[Real, MAXWELL_NC]:
    """Ghost-state synthesis. `face_axis` ∈ {0, 1, 2} selects the wall normal.

    `BC_PEC_WALL`: flip the two tangential E components and the normal B.
    `BC_PMC_WALL`: dual — flip the normal E and the two tangential B.
    Anything else copies the interior state.
    """
    var q: InlineArray[Real, MAXWELL_NC] = q_interior
    if bc_kind == BC_PEC_WALL:
        if face_axis == 0:
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BX] = -q_interior[IDX_BX]
        elif face_axis == 1:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BY] = -q_interior[IDX_BY]
        else:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_BZ] = -q_interior[IDX_BZ]
    elif bc_kind == BC_PMC_WALL:
        if face_axis == 0:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_BY] = -q_interior[IDX_BY]
            q[IDX_BZ] = -q_interior[IDX_BZ]
        elif face_axis == 1:
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_BX] = -q_interior[IDX_BX]
            q[IDX_BZ] = -q_interior[IDX_BZ]
        else:
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BX] = -q_interior[IDX_BX]
            q[IDX_BY] = -q_interior[IDX_BY]
    return q


def _maxwell_rhs_kernel_3d[
    P: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    c_light: Real,
    jacobian_x: Real, jacobian_y: Real, jacobian_z: Real,
    num_elements: Int,
):
    @parameter
    def flux_x(qn: InlineArray[Real, MAXWELL_NC]) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_x_static(qn, c_light)

    @parameter
    def flux_y(qn: InlineArray[Real, MAXWELL_NC]) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_y_static(qn, c_light)

    @parameter
    def flux_z(qn: InlineArray[Real, MAXWELL_NC]) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_z_static(qn, c_light)

    @parameter
    def num_flux(
        q_l: InlineArray[Real, MAXWELL_NC],
        q_r: InlineArray[Real, MAXWELL_NC],
        nx: Real, ny: Real, nz: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.numerical_flux_static(q_l, q_r, nx, ny, nz, c_light)

    @parameter
    def apply_bc(
        q_interior: InlineArray[Real, MAXWELL_NC],
        bc_kind: Int, face_axis: Int,
    ) -> InlineArray[Real, MAXWELL_NC]:
        return _apply_bc_maxwell_3d(q_interior, bc_kind, face_axis)

    _dg_3d_compute_rhs_core[
        P, MAXWELL_NC, flux_x, flux_y, flux_z, num_flux, apply_bc,
        QLayout, RhsLayout, FaceLayout, EFLayout,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        jacobian_x, jacobian_y, jacobian_z, num_elements,
    )


struct DGMaxwellSolver3D[P: Int]:
    """Host-side launcher for the 3D Maxwell DG kernel."""

    var app: Maxwell

    def __init__(out self, app: Maxwell):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, MAXWELL_NC],
        mut rhs: DGState3D[Self.P, MAXWELL_NC],
    ) raises:
        comptime P1 = Self.P + 1

        var q_layout = row_major(Idx(q.buf_len))
        var face_layout = row_major(Idx(mesh.num_faces))
        var ef_layout = row_major(Idx(mesh.num_elements * 6))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _maxwell_rhs_kernel_3d[
            Self.P, type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.c_light,
            mesh.jacobian_x, mesh.jacobian_y, mesh.jacobian_z,
            mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=(P1, P1, P1),
        )
