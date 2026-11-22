# 3D nodal DG spatial solver for MixedGLM ideal MHD — thin wrapper over
# the shared `_dg_3d_compute_rhs_core` template. Supports Rusanov + HLLD
# (USE_HLLD parameter). Parabolic ψ damping is applied separately by the
# stepper via `apply_glm_damping_3d`.

from std.collections import Array
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_core_3d import (
    _dg_3d_compute_rhs_core, DG_CORE_3D_ELEMS_PER_BLOCK,
)
from warpxm.variable.state_3d import DGState3D


@always_inline
def _apply_bc_imhd_glm_3d(
    q_interior: Array[Real, MHD_GLM_NC],
    bc_kind: Int,
    face_axis: Int,
) -> Array[Real, MHD_GLM_NC]:
    """Copyout ghost state for GLM-MHD (no walls supported yet)."""
    return q_interior.copy()


def _imhd_glm_rhs_kernel_3d[
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
    gamma: Real, rho_min: Real, press_min: Real, c_h: Real,
    inv_jacobian_x: Real, inv_jacobian_y: Real, inv_jacobian_z: Real,
    num_elements_i64: Int64,
):
    var num_elements = Int(num_elements_i64)
    def flux_x(qn: Array[Real, MHD_GLM_NC]) {imm} -> Array[Real, MHD_GLM_NC]:
        return ImhdGLM.internal_flux_x_static(qn, gamma, rho_min, press_min, c_h)

    def flux_y(qn: Array[Real, MHD_GLM_NC]) {imm} -> Array[Real, MHD_GLM_NC]:
        return ImhdGLM.internal_flux_y_static(qn, gamma, rho_min, press_min, c_h)

    def flux_z(qn: Array[Real, MHD_GLM_NC]) {imm} -> Array[Real, MHD_GLM_NC]:
        return ImhdGLM.internal_flux_z_static(qn, gamma, rho_min, press_min, c_h)

    def num_flux(
        q_l: Array[Real, MHD_GLM_NC],
        q_r: Array[Real, MHD_GLM_NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, MHD_GLM_NC]:
        comptime if USE_HLLD:
            return ImhdGLM.numerical_flux_hlld_static(
                q_l, q_r, nx, ny, nz, gamma, rho_min, press_min, c_h,
            )
        else:
            return ImhdGLM.numerical_flux_static(
                q_l, q_r, nx, ny, nz, gamma, rho_min, press_min, c_h,
            )

    def apply_bc(
        q_interior: Array[Real, MHD_GLM_NC],
        bc_kind: Int, face_axis: Int,
    ) {imm} -> Array[Real, MHD_GLM_NC]:
        return _apply_bc_imhd_glm_3d(q_interior, bc_kind, face_axis)

    _dg_3d_compute_rhs_core[
        P, MHD_GLM_NC,
        QLayout, RhsLayout, FaceLayout, EFLayout,
        DG_CORE_3D_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian_x, inv_jacobian_y, inv_jacobian_z, num_elements,
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGImhdGLMSolver3D[P: Int, USE_HLLD: Bool = False]:
    """Host-side launcher for the 3D MixedGLM-MHD DG kernel.

    `USE_HLLD = False` (default) uses Rusanov with c_max bumped to `c_h`;
    `USE_HLLD = True` uses IMHD HLLD for the 8 MHD components and the
    exact (B_n, ψ) 2x2 Riemann block on top.
    """

    var app: ImhdGLM

    def __init__(out self, app: ImhdGLM):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, MHD_GLM_NC],
        mut rhs: DGState3D[Self.P, MHD_GLM_NC],
    ) raises:
        comptime P1 = Self.P + 1

        var q_layout = row_major(Int(q.buf_len))
        var face_layout = row_major(Int(mesh.num_faces))
        var ef_layout = row_major(Int(mesh.num_elements * 6))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _imhd_glm_rhs_kernel_3d[
            Self.P, Self.USE_HLLD,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        comptime EB = DG_CORE_3D_ELEMS_PER_BLOCK
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            self.app.c_h,
            Real(1.0) / mesh.jacobian_x, Real(1.0) / mesh.jacobian_y, Real(1.0) / mesh.jacobian_z,
            Int64(mesh.num_elements),
            grid_dim=ceildiv(mesh.num_elements, EB),
            block_dim=(P1, P1, P1 * EB),
        )
