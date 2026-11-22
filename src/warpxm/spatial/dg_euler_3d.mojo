# 3D nodal DG spatial solver for 5-moment Euler — thin wrapper over the
# shared `_dg_3d_compute_rhs_core` template. Mirrors `dg_euler_2d.mojo`.

from std.collections import Array
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.euler import Euler, EULER_NC, IDX_MX, IDX_MY, IDX_MZ
from warpxm.core.bc import BC_WALL_FREESLIP
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_core_3d import (
    _dg_3d_compute_rhs_core, DG_CORE_3D_ELEMS_PER_BLOCK,
)
from warpxm.variable.state_3d import DGState3D


@always_inline
def _apply_bc_euler_3d(
    q_interior: Array[Real, EULER_NC],
    bc_kind: Int,
    face_axis: Int,
) -> Array[Real, EULER_NC]:
    """Ghost-state synthesis. `face_axis` ∈ {0, 1, 2} selects which
    momentum component flips for BC_WALL_FREESLIP. Other BC kinds fall
    back to copy-out."""
    var q: Array[Real, EULER_NC] = q_interior.copy()
    if bc_kind == BC_WALL_FREESLIP:
        if face_axis == 0:
            q[IDX_MX] = -q_interior[IDX_MX]
        elif face_axis == 1:
            q[IDX_MY] = -q_interior[IDX_MY]
        else:
            q[IDX_MZ] = -q_interior[IDX_MZ]
    return q^


def _euler_rhs_kernel_3d[
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
    gamma: Real, rho_min: Real, press_min: Real,
    inv_jacobian_x: Real, inv_jacobian_y: Real, inv_jacobian_z: Real,
    num_elements_i64: Int64,
):
    var num_elements = Int(num_elements_i64)
    def flux_x(qn: Array[Real, EULER_NC]) {imm} -> Array[Real, EULER_NC]:
        return Euler.internal_flux_x_static(qn, gamma, rho_min, press_min)

    def flux_y(qn: Array[Real, EULER_NC]) {imm} -> Array[Real, EULER_NC]:
        return Euler.internal_flux_y_static(qn, gamma, rho_min, press_min)

    def flux_z(qn: Array[Real, EULER_NC]) {imm} -> Array[Real, EULER_NC]:
        return Euler.internal_flux_z_static(qn, gamma, rho_min, press_min)

    def num_flux(
        q_l: Array[Real, EULER_NC],
        q_r: Array[Real, EULER_NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, EULER_NC]:
        return Euler.numerical_flux_static(
            q_l, q_r, nx, ny, nz, gamma, rho_min, press_min,
        )

    def apply_bc(
        q_interior: Array[Real, EULER_NC],
        bc_kind: Int, face_axis: Int,
    ) {imm} -> Array[Real, EULER_NC]:
        return _apply_bc_euler_3d(q_interior, bc_kind, face_axis)

    _dg_3d_compute_rhs_core[
        P, EULER_NC,
        QLayout, RhsLayout, FaceLayout, EFLayout,
        DG_CORE_3D_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian_x, inv_jacobian_y, inv_jacobian_z, num_elements,
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGEulerSolver3D[P: Int]:
    """Host-side launcher for the 3D 5-moment Euler DG kernel."""

    var app: Euler

    def __init__(out self, app: Euler):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, EULER_NC],
        mut rhs: DGState3D[Self.P, EULER_NC],
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

        comptime kernel = _euler_rhs_kernel_3d[
            Self.P, type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        comptime EB = DG_CORE_3D_ELEMS_PER_BLOCK
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            Real(1.0) / mesh.jacobian_x, Real(1.0) / mesh.jacobian_y, Real(1.0) / mesh.jacobian_z,
            Int64(mesh.num_elements),
            grid_dim=ceildiv(mesh.num_elements, EB),
            block_dim=(P1, P1, P1 * EB),
        )
