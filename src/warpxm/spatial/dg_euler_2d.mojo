# 2D nodal DG spatial solver for 5-moment Euler.
#
# A thin wrapper around the shared `_dg_2d_compute_rhs_core` kernel core.
# The wrapper kernel `_euler_rhs_kernel_2d` exposes the Euler-specific
# scalar parameters (γ, ρ_min, P_min) at the GPU launch boundary and
# defines four `def ... {imm}` closures (`flux_x`, `flux_y`, `num_flux`,
# `apply_bc`) that capture those scalars and dispatch to the app's
# static flux methods.

from std.collections import Array
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.euler import Euler, EULER_NC, IDX_MX, IDX_MY
from warpxm.core.bc import BC_WALL_FREESLIP
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from std.math import ceildiv
from warpxm.spatial.dg_core_2d import (
    _dg_2d_compute_rhs_core, DG_CORE_2D_ELEMS_PER_BLOCK,
)
from warpxm.variable.state_2d import DGState2D


@always_inline
def _apply_bc_euler(
    q_interior: Array[Real, EULER_NC],
    bc_kind: Int,
    face_axis: Int,
) -> Array[Real, EULER_NC]:
    """Synthesize the ghost-side Euler state from an interior state.

    `face_axis` is 0 for ξ-axis (normal ±x̂) faces, 1 for η-axis (normal ±ŷ).
    For a free-slip wall the normal momentum component is reversed and
    everything else is copied; copyout just copies.
    """
    var q: Array[Real, EULER_NC] = q_interior.copy()
    if bc_kind == BC_WALL_FREESLIP:
        if face_axis == 0:
            q[IDX_MX] = -q_interior[IDX_MX]
        else:
            q[IDX_MY] = -q_interior[IDX_MY]
    return q^


def _euler_rhs_kernel_2d[
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
    gamma: Real,
    rho_min: Real,
    press_min: Real,
    jacobian_x: Real,
    jacobian_y: Real,
    num_elements_i64: Int64,
):
    var num_elements = Int(num_elements_i64)
    def flux_x(qn: Array[Real, EULER_NC]) {imm} -> Array[Real, EULER_NC]:
        return Euler.internal_flux_x_static(qn, gamma, rho_min, press_min)

    def flux_y(qn: Array[Real, EULER_NC]) {imm} -> Array[Real, EULER_NC]:
        return Euler.internal_flux_y_static(qn, gamma, rho_min, press_min)

    def num_flux(
        q_l: Array[Real, EULER_NC],
        q_r: Array[Real, EULER_NC],
        nx: Real, ny: Real,
    ) {imm} -> Array[Real, EULER_NC]:
        return Euler.numerical_flux_static(
            q_l, q_r, nx, ny, Real(0.0), gamma, rho_min, press_min,
        )

    def apply_bc(
        q_interior: Array[Real, EULER_NC],
        bc_kind: Int, face_axis: Int,
    ) {imm} -> Array[Real, EULER_NC]:
        return _apply_bc_euler(q_interior, bc_kind, face_axis)

    _dg_2d_compute_rhs_core[
        P, EULER_NC,
        QLayout, RhsLayout, FaceLayout, EFLayout,
        DG_CORE_2D_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        jacobian_x, jacobian_y, num_elements,
        flux_x, flux_y, num_flux, apply_bc,
    )


struct DGEulerSolver2D[P: Int]:
    """Host-side launcher for the 2D 5-moment Euler DG kernel."""

    var app: Euler

    def __init__(out self, app: Euler):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, EULER_NC],
        mut rhs: DGState2D[Self.P, EULER_NC],
    ) raises:
        comptime P1 = Self.P + 1

        var q_layout = row_major(Int(q.buf_len))
        var face_layout = row_major(Int(mesh.num_faces))
        var ef_layout = row_major(Int(mesh.num_elements * 4))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _euler_rhs_kernel_2d[
            Self.P,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        comptime EB = DG_CORE_2D_ELEMS_PER_BLOCK
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            mesh.jacobian_x, mesh.jacobian_y, Int64(mesh.num_elements),
            grid_dim=ceildiv(mesh.num_elements, EB),
            block_dim=(P1, P1, EB),
        )
