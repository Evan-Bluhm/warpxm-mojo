# 1D nodal DG spatial solver for 5-moment Euler — thin wrapper over the
# shared `_dg_1d_compute_rhs_core` template.

from std.collections import Array
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.euler import Euler, EULER_NC, IDX_MX
from warpxm.core.bc import BC_WALL_FREESLIP
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_core_1d import (
    _dg_1d_compute_rhs_core, DG_CORE_1D_ELEMS_PER_BLOCK,
)
from warpxm.variable.state_1d import DGState1D


@always_inline
def _apply_bc_euler_1d(
    q_interior: Array[Real, EULER_NC], bc_kind: Int,
) -> Array[Real, EULER_NC]:
    """Free-slip wall flips the x-momentum; everything else copies."""
    var q: Array[Real, EULER_NC] = q_interior.copy()
    if bc_kind == BC_WALL_FREESLIP:
        q[IDX_MX] = -q_interior[IDX_MX]
    return q^


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
    inv_jacobian: Real,
    num_elements_i64: Int64,
):
    var num_elements = Int(num_elements_i64)
    def flux_x(qn: Array[Real, EULER_NC]) {imm} -> Array[Real, EULER_NC]:
        return Euler.internal_flux_x_static(qn, gamma, rho_min, press_min)

    def num_flux(
        q_l: Array[Real, EULER_NC], q_r: Array[Real, EULER_NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, EULER_NC]:
        return Euler.numerical_flux_static(
            q_l, q_r, nx, ny, nz, gamma, rho_min, press_min,
        )

    def apply_bc(
        q_interior: Array[Real, EULER_NC], bc_kind: Int,
    ) {imm} -> Array[Real, EULER_NC]:
        return _apply_bc_euler_1d(q_interior, bc_kind)

    _dg_1d_compute_rhs_core[
        P, EULER_NC,
        QLayout, RhsLayout, ElemFaceLayout,
        DG_CORE_1D_ELEMS_PER_BLOCK,
    ](
        q, rhs, elem_left_face, elem_right_face,
        face_left, face_right, face_bc,
        inv_jacobian, num_elements,
        flux_x, num_flux, apply_bc,
    )


struct DGEulerSolver1D[P: Int]:
    """Host-side launcher for the 1D 5-moment Euler DG kernel."""

    var app: Euler

    def __init__(out self, app: Euler):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, EULER_NC],
        mut rhs: DGState1D[Self.P, EULER_NC],
    ) raises:
        comptime num_nodes_per_elem = Self.P + 1

        var q_layout = row_major(Int(q.buf_len))
        var face_layout = row_major(Int(mesh.num_faces))
        var elem_face_layout = row_major(Int(mesh.num_elements))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var elf_view = TileTensor(mesh.elem_left_face_buf, elem_face_layout)
        var erf_view = TileTensor(mesh.elem_right_face_buf, elem_face_layout)
        var fl_view = TileTensor(mesh.face_left_buf, face_layout)
        var fr_view = TileTensor(mesh.face_right_buf, face_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _euler_rhs_kernel_1d[
            Self.P,
            type_of(q_layout), type_of(q_layout), type_of(elem_face_layout),
        ]
        comptime EB = DG_CORE_1D_ELEMS_PER_BLOCK
        ctx.enqueue_function[kernel](
            q_view, rhs_view, elf_view, erf_view, fl_view, fr_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            Real(1.0) / mesh.jacobian, Int64(mesh.num_elements),
            grid_dim=ceildiv(mesh.num_elements, EB),
            block_dim=(num_nodes_per_elem, EB),
        )
