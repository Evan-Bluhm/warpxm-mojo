# 3D nodal DG spatial solver for Ideal MHD — thin wrapper over the
# shared `_dg_3d_compute_rhs_core` template. Supports Rusanov + HLLD
# numerical fluxes (USE_HLLD parameter on the solver struct).

from std.collections import Array
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_MX, IDX_MY, IDX_MZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.bc import BC_WALL_FREESLIP, BC_PEC_WALL
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_core_3d import (
    _dg_3d_compute_rhs_core, DG_CORE_3D_ELEMS_PER_BLOCK,
)
from warpxm.variable.state_3d import DGState3D


@always_inline
def _apply_bc_imhd_3d(
    q_interior: Array[Real, MHD_NC],
    bc_kind: Int,
    face_axis: Int,
) -> Array[Real, MHD_NC]:
    """Ghost-state synthesis.

    * `BC_WALL_FREESLIP` flips the normal momentum component only.
    * `BC_PEC_WALL` (perfectly conducting wall) flips normal momentum
      AND normal B — enforcing impenetrability and B_n = 0 on the wall.
    * Anything else (copyout) → ghost = interior.
    """
    var q: Array[Real, MHD_NC] = q_interior.copy()
    if bc_kind == BC_WALL_FREESLIP:
        if face_axis == 0:
            q[IDX_MX] = -q_interior[IDX_MX]
        elif face_axis == 1:
            q[IDX_MY] = -q_interior[IDX_MY]
        else:
            q[IDX_MZ] = -q_interior[IDX_MZ]
    elif bc_kind == BC_PEC_WALL:
        if face_axis == 0:
            q[IDX_MX] = -q_interior[IDX_MX]
            q[IDX_BX] = -q_interior[IDX_BX]
        elif face_axis == 1:
            q[IDX_MY] = -q_interior[IDX_MY]
            q[IDX_BY] = -q_interior[IDX_BY]
        else:
            q[IDX_MZ] = -q_interior[IDX_MZ]
            q[IDX_BZ] = -q_interior[IDX_BZ]
    return q^


def _imhd_rhs_kernel_3d[
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
    gamma: Real, rho_min: Real, press_min: Real,
    jacobian_x: Real, jacobian_y: Real, jacobian_z: Real,
    num_elements_i64: Int64,
):
    var num_elements = Int(num_elements_i64)
    def flux_x(qn: Array[Real, MHD_NC]) {imm} -> Array[Real, MHD_NC]:
        return IMHD.internal_flux_x_static(qn, gamma, rho_min, press_min)

    def flux_y(qn: Array[Real, MHD_NC]) {imm} -> Array[Real, MHD_NC]:
        return IMHD.internal_flux_y_static(qn, gamma, rho_min, press_min)

    def flux_z(qn: Array[Real, MHD_NC]) {imm} -> Array[Real, MHD_NC]:
        return IMHD.internal_flux_z_static(qn, gamma, rho_min, press_min)

    def num_flux(
        q_l: Array[Real, MHD_NC],
        q_r: Array[Real, MHD_NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, MHD_NC]:
        comptime if USE_HLLD:
            return IMHD.numerical_flux_hlld_static(
                q_l, q_r, nx, ny, nz, gamma, rho_min, press_min,
            )
        else:
            return IMHD.numerical_flux_static(
                q_l, q_r, nx, ny, nz, gamma, rho_min, press_min,
            )

    def apply_bc(
        q_interior: Array[Real, MHD_NC],
        bc_kind: Int, face_axis: Int,
    ) {imm} -> Array[Real, MHD_NC]:
        return _apply_bc_imhd_3d(q_interior, bc_kind, face_axis)

    _dg_3d_compute_rhs_core[
        P, MHD_NC,
        QLayout, RhsLayout, FaceLayout, EFLayout,
        DG_CORE_3D_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        jacobian_x, jacobian_y, jacobian_z, num_elements,
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGImhdSolver3D[P: Int, USE_HLLD: Bool = False]:
    """Host-side launcher for the 3D ideal-MHD DG kernel.

    `USE_HLLD = False` (default) uses Rusanov / Lax-Friedrichs.
    `USE_HLLD = True` switches to Miyoshi-Kusano HLLD for all
    axis-aligned faces.
    """

    var app: IMHD

    def __init__(out self, app: IMHD):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, MHD_NC],
        mut rhs: DGState3D[Self.P, MHD_NC],
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

        comptime kernel = _imhd_rhs_kernel_3d[
            Self.P, Self.USE_HLLD,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        comptime EB = DG_CORE_3D_ELEMS_PER_BLOCK
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            mesh.jacobian_x, mesh.jacobian_y, mesh.jacobian_z,
            Int64(mesh.num_elements),
            grid_dim=ceildiv(mesh.num_elements, EB),
            block_dim=(P1, P1, P1 * EB),
        )
