# 2D nodal DG spatial solver for linear advection.
#
# Thin wrapper over the shared `_dg_2d_compute_rhs_core` kernel core.
# `NC`-parameterised over the tracer count — every component advects
# with the same uniform velocity. Used with periodic meshes in the
# test suite; the `apply_bc` closure is a copyout fallback for
# completeness.

from std.collections import InlineArray
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.advection import Advection
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_core_2d import _dg_2d_compute_rhs_core
from warpxm.variable.state_2d import DGState2D


@always_inline
def _apply_bc_advection_2d[NC: Int](
    q_interior: InlineArray[Real, NC],
    bc_kind: Int,
    face_axis: Int,
) -> InlineArray[Real, NC]:
    """Copyout fallback. Advection 2D currently only runs on periodic
    meshes in our test suite, so this is just a placeholder."""
    return q_interior


def _advection_rhs_kernel_2d[
    P: Int, NC: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    velocity_x: Real,
    velocity_y: Real,
    jacobian_x: Real,
    jacobian_y: Real,
    num_elements: Int,
):
    @parameter
    def flux_x(qn: InlineArray[Real, NC]) -> InlineArray[Real, NC]:
        return Advection[NC].internal_flux_x_static(qn, velocity_x)

    @parameter
    def flux_y(qn: InlineArray[Real, NC]) -> InlineArray[Real, NC]:
        return Advection[NC].internal_flux_y_static(qn, velocity_y)

    @parameter
    def num_flux(
        q_l: InlineArray[Real, NC],
        q_r: InlineArray[Real, NC],
        nx: Real, ny: Real,
    ) -> InlineArray[Real, NC]:
        return Advection[NC].numerical_flux_static(
            q_l, q_r, nx, ny, Real(0.0),
            velocity_x, velocity_y, Real(0.0),
        )

    @parameter
    def apply_bc(
        q_interior: InlineArray[Real, NC],
        bc_kind: Int, face_axis: Int,
    ) -> InlineArray[Real, NC]:
        return _apply_bc_advection_2d[NC](q_interior, bc_kind, face_axis)

    _dg_2d_compute_rhs_core[
        P, NC, flux_x, flux_y, num_flux, apply_bc,
        QLayout, RhsLayout, FaceLayout, EFLayout,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        jacobian_x, jacobian_y, num_elements,
    )


struct DGSpatialSolver2D[P: Int, NC: Int]:
    """Host-side launcher for the 2D advection DG kernel."""

    var app: Advection[Self.NC]

    def __init__(out self, app: Advection[Self.NC]):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, Self.NC],
        mut rhs: DGState2D[Self.P, Self.NC],
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

        comptime kernel = _advection_rhs_kernel_2d[
            Self.P, Self.NC,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.velocity_x, self.app.velocity_y,
            mesh.jacobian_x, mesh.jacobian_y, mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=(P1, P1),
        )
