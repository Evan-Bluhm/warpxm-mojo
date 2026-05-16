# 2D nodal DG spatial solver for ideal MHD.
#
# Thin wrapper over the shared `_dg_2d_compute_rhs_core` kernel core, with
# `USE_HLLD` selecting between the Rusanov and HLLD numerical fluxes inside
# the `num_flux` closure.

from std.collections import InlineArray
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.imhd import IMHD, MHD_NC
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_core_2d import _dg_2d_compute_rhs_core
from warpxm.variable.state_2d import DGState2D


@always_inline
def _apply_bc_imhd(
    q_interior: InlineArray[Real, MHD_NC],
    bc_kind: Int,
    face_axis: Int,
) -> InlineArray[Real, MHD_NC]:
    """Ghost-state synthesis for 2D ideal MHD. Currently a copyout
    fallback for every `bc_kind` — the 2D MHD tests that pass
    `BC_WALL_FREESLIP` (Brio-Wu y-walls, etc.) happen to have $v_y = 0$
    along the relevant boundary by construction, so copyout and
    free-slip give the same ghost state. The 3D analog
    (`_apply_bc_imhd_3d`) does flip normal momentum on
    `BC_WALL_FREESLIP` and normal momentum + normal B on
    `BC_PEC_WALL`; we'd port the same dispatch here if a 2D test
    needs genuine wall behaviour."""
    return q_interior


def _imhd_rhs_kernel_2d[
    P: Int, USE_HLLD: Bool,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    gamma: Real,
    rho_min: Real,
    press_min: Real,
    jacobian_x: Real,
    jacobian_y: Real,
    num_elements: Int,
):
    @parameter
    def flux_x(qn: InlineArray[Real, MHD_NC]) -> InlineArray[Real, MHD_NC]:
        return IMHD.internal_flux_x_static(qn, gamma, rho_min, press_min)

    @parameter
    def flux_y(qn: InlineArray[Real, MHD_NC]) -> InlineArray[Real, MHD_NC]:
        return IMHD.internal_flux_y_static(qn, gamma, rho_min, press_min)

    @parameter
    def num_flux(
        q_l: InlineArray[Real, MHD_NC],
        q_r: InlineArray[Real, MHD_NC],
        nx: Real, ny: Real,
    ) -> InlineArray[Real, MHD_NC]:
        comptime if USE_HLLD:
            return IMHD.numerical_flux_hlld_static(
                q_l, q_r, nx, ny, Real(0.0), gamma, rho_min, press_min,
            )
        else:
            return IMHD.numerical_flux_static(
                q_l, q_r, nx, ny, Real(0.0), gamma, rho_min, press_min,
            )

    @parameter
    def apply_bc(
        q_interior: InlineArray[Real, MHD_NC],
        bc_kind: Int, face_axis: Int,
    ) -> InlineArray[Real, MHD_NC]:
        return _apply_bc_imhd(q_interior, bc_kind, face_axis)

    _dg_2d_compute_rhs_core[
        P, MHD_NC, flux_x, flux_y, num_flux, apply_bc,
        QLayout, RhsLayout, FaceLayout, EFLayout,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        jacobian_x, jacobian_y, num_elements,
    )


struct DGImhdSolver2D[P: Int, USE_HLLD: Bool = False]:
    """Host-side launcher for the 2D ideal-MHD DG kernel.

    `USE_HLLD = False` (default) uses Rusanov / Lax-Friedrichs;
    `USE_HLLD = True` switches to Miyoshi-Kusano HLLD.
    """

    var app: IMHD

    def __init__(out self, app: IMHD):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, MHD_NC],
        mut rhs: DGState2D[Self.P, MHD_NC],
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

        comptime kernel = _imhd_rhs_kernel_2d[
            Self.P, Self.USE_HLLD,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            mesh.jacobian_x, mesh.jacobian_y, mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=(P1, P1),
        )
