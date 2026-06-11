# 2D nodal DG spatial solver for Maxwell's equations.
#
# Thin wrapper over the shared `_dg_2d_compute_rhs_core` kernel core.
# Maxwell's only physics parameter is `c_light`; the BC handler supports
# PEC and PMC walls in addition to copyout.

from std.collections import InlineArray
from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.bc import BC_PEC_WALL, BC_PMC_WALL
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_core_2d import (
    _dg_2d_compute_rhs_core, DG_CORE_2D_ELEMS_PER_BLOCK,
)
from warpxm.variable.state_2d import DGState2D


@always_inline
def _apply_bc_maxwell(
    q_interior: InlineArray[Real, MAXWELL_NC],
    bc_kind: Int,
    face_axis: Int,
) -> InlineArray[Real, MAXWELL_NC]:
    """Synthesize the ghost-side Maxwell state from an interior state.

    `face_axis` is 0 for ξ-axis (normal ±x̂) faces, 1 for η-axis (normal ±ŷ).

    * `BC_PEC_WALL` (perfect electric conductor): tangential E and normal B
      get reversed in the ghost cell so the face-average vanishes for the
      "should-be-zero" components.
    * `BC_PMC_WALL` (perfect magnetic conductor): the dual — normal E and
      tangential B get reversed.
    * Anything else → copyout.
    """
    var q: InlineArray[Real, MAXWELL_NC] = q_interior
    if bc_kind == BC_PEC_WALL:
        if face_axis == 0:
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BX] = -q_interior[IDX_BX]
        else:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BY] = -q_interior[IDX_BY]
    elif bc_kind == BC_PMC_WALL:
        if face_axis == 0:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_BY] = -q_interior[IDX_BY]
            q[IDX_BZ] = -q_interior[IDX_BZ]
        else:
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_BX] = -q_interior[IDX_BX]
            q[IDX_BZ] = -q_interior[IDX_BZ]
    return q


def _maxwell_rhs_kernel_2d[
    P: Int,
    QLayout: TensorLayout,
    RhsLayout: TensorLayout,
    FaceLayout: TensorLayout,
    EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    c_light: Real,
    jacobian_x: Real,
    jacobian_y: Real,
    num_elements: Int,
):
    @parameter
    def flux_x(qn: InlineArray[Real, MAXWELL_NC]) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_x_static(qn, c_light)

    @parameter
    def flux_y(qn: InlineArray[Real, MAXWELL_NC]) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_y_static(qn, c_light)

    @parameter
    def num_flux(
        q_l: InlineArray[Real, MAXWELL_NC],
        q_r: InlineArray[Real, MAXWELL_NC],
        nx: Real, ny: Real,
    ) -> InlineArray[Real, MAXWELL_NC]:
        return Maxwell.numerical_flux_static(q_l, q_r, nx, ny, Real(0.0), c_light)

    @parameter
    def apply_bc(
        q_interior: InlineArray[Real, MAXWELL_NC],
        bc_kind: Int, face_axis: Int,
    ) -> InlineArray[Real, MAXWELL_NC]:
        return _apply_bc_maxwell(q_interior, bc_kind, face_axis)

    _dg_2d_compute_rhs_core[
        P, MAXWELL_NC, flux_x, flux_y, num_flux, apply_bc,
        QLayout, RhsLayout, FaceLayout, EFLayout,
        DG_CORE_2D_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        jacobian_x, jacobian_y, num_elements,
    )


struct DGMaxwellSolver2D[P: Int]:
    """Host-side launcher for the 2D Maxwell DG kernel."""

    var app: Maxwell

    def __init__(out self, app: Maxwell):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, MAXWELL_NC],
        mut rhs: DGState2D[Self.P, MAXWELL_NC],
    ) raises:
        comptime P1 = Self.P + 1

        var q_layout = row_major(Idx(q.buf_len))
        var face_layout = row_major(Idx(mesh.num_faces))
        var ef_layout = row_major(Idx(mesh.num_elements * 4))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _maxwell_rhs_kernel_2d[
            Self.P,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        comptime EB = DG_CORE_2D_ELEMS_PER_BLOCK
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.c_light,
            mesh.jacobian_x, mesh.jacobian_y, mesh.num_elements,
            grid_dim=ceildiv(mesh.num_elements, EB),
            block_dim=(P1, P1, EB),
        )
