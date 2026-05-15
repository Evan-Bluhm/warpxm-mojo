# 2D nodal DG spatial solver for MixedGLM ideal MHD (9 components).
#
# Thin wrapper over the shared `_dg_2d_compute_rhs_core` kernel core.
# The hyperbolic ψ-coupling is built into the flux tensor (extra `c_h`
# parameter on the flux methods); the parabolic damping source on ψ
# lives in `imhd_glm_source_2d.mojo` and is applied as a post-pass in
# the stepper.

from std.collections import InlineArray
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.imhd_glm_2d import ImhdGLM2D, MHD_GLM_NC
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_core_2d import _dg_2d_compute_rhs_core
from warpxm.variable.state_2d import DGState2D


@always_inline
def _apply_bc_imhd_glm(
    q_interior: InlineArray[Real, MHD_GLM_NC],
    bc_kind: Int,
    face_axis: Int,
) -> InlineArray[Real, MHD_GLM_NC]:
    """Copyout-style ghost state for GLM-MHD."""
    return q_interior


def _imhd_glm_rhs_kernel_2d[
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
    gamma: Real,
    rho_min: Real,
    press_min: Real,
    c_h: Real,
    jacobian_x: Real,
    jacobian_y: Real,
    num_elements: Int,
):
    @parameter
    def flux_x(qn: InlineArray[Real, MHD_GLM_NC]) -> InlineArray[Real, MHD_GLM_NC]:
        return ImhdGLM2D.internal_flux_x_static(qn, gamma, rho_min, press_min, c_h)

    @parameter
    def flux_y(qn: InlineArray[Real, MHD_GLM_NC]) -> InlineArray[Real, MHD_GLM_NC]:
        return ImhdGLM2D.internal_flux_y_static(qn, gamma, rho_min, press_min, c_h)

    @parameter
    def num_flux(
        q_l: InlineArray[Real, MHD_GLM_NC],
        q_r: InlineArray[Real, MHD_GLM_NC],
        nx: Real, ny: Real,
    ) -> InlineArray[Real, MHD_GLM_NC]:
        return ImhdGLM2D.numerical_flux_static(
            q_l, q_r, nx, ny, gamma, rho_min, press_min, c_h,
        )

    @parameter
    def apply_bc(
        q_interior: InlineArray[Real, MHD_GLM_NC],
        bc_kind: Int, face_axis: Int,
    ) -> InlineArray[Real, MHD_GLM_NC]:
        return _apply_bc_imhd_glm(q_interior, bc_kind, face_axis)

    _dg_2d_compute_rhs_core[
        P, MHD_GLM_NC, flux_x, flux_y, num_flux, apply_bc,
        QLayout, RhsLayout, FaceLayout, EFLayout,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        jacobian_x, jacobian_y, num_elements,
    )


struct DGImhdGLMSolver2D[P: Int]:
    """Host-side launcher for the 2D GLM-MHD DG kernel."""

    var app: ImhdGLM2D

    def __init__(out self, app: ImhdGLM2D):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, MHD_GLM_NC],
        mut rhs: DGState2D[Self.P, MHD_GLM_NC],
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

        comptime kernel = _imhd_glm_rhs_kernel_2d[
            Self.P,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            self.app.c_h,
            mesh.jacobian_x, mesh.jacobian_y, mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=(P1, P1),
        )
