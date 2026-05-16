# 1D nodal DG spatial solver for advection — thin wrapper over the
# shared `_dg_1d_compute_rhs_core` template. NC-parameterised over the
# tracer count.

from std.collections import InlineArray
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.advection import Advection
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_core_1d import _dg_1d_compute_rhs_core
from warpxm.variable.state_1d import DGState1D


@always_inline
def _apply_bc_scalar[NC: Int](
    q_interior: InlineArray[Real, NC], bc_kind: Int,
) -> InlineArray[Real, NC]:
    """Ghost-state for a passive scalar — copyout regardless of BC kind."""
    return q_interior


def _advection_rhs_kernel[
    P: Int, NC: Int,
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
    velocity: Real,
    jacobian: Real,
    num_elements: Int,
):
    @parameter
    def flux_x(qn: InlineArray[Real, NC]) -> InlineArray[Real, NC]:
        return Advection[NC].internal_flux_x_static(qn, velocity)

    @parameter
    def num_flux(
        q_l: InlineArray[Real, NC], q_r: InlineArray[Real, NC],
        nx: Real, ny: Real, nz: Real,
    ) -> InlineArray[Real, NC]:
        return Advection[NC].numerical_flux_static(
            q_l, q_r, nx, ny, nz, velocity, Real(0.0), Real(0.0),
        )

    @parameter
    def apply_bc(
        q_interior: InlineArray[Real, NC], bc_kind: Int,
    ) -> InlineArray[Real, NC]:
        return _apply_bc_scalar[NC](q_interior, bc_kind)

    _dg_1d_compute_rhs_core[
        P, NC, flux_x, num_flux, apply_bc,
        QLayout, RhsLayout, ElemFaceLayout,
    ](
        q, rhs, elem_left_face, elem_right_face,
        face_left, face_right, face_bc,
        jacobian, num_elements,
    )


struct DGSpatialSolver1D[P: Int, NC: Int]:
    """Host-side launcher for the multi-component 1D advection DG kernel."""

    var app: Advection[Self.NC]

    def __init__(out self, app: Advection[Self.NC]):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, Self.NC],
        mut rhs: DGState1D[Self.P, Self.NC],
    ) raises:
        comptime num_nodes_per_elem = Self.P + 1

        var q_layout = row_major(Idx(q.buf_len))
        var face_layout = row_major(Idx(mesh.num_faces))
        var elem_face_layout = row_major(Idx(mesh.num_elements))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var elf_view = TileTensor(mesh.elem_left_face_buf, elem_face_layout)
        var erf_view = TileTensor(mesh.elem_right_face_buf, elem_face_layout)
        var fl_view = TileTensor(mesh.face_left_buf, face_layout)
        var fr_view = TileTensor(mesh.face_right_buf, face_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _advection_rhs_kernel[
            Self.P, Self.NC,
            type_of(q_layout), type_of(q_layout), type_of(elem_face_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, elf_view, erf_view, fl_view, fr_view, fbc_view,
            self.app.velocity_x, mesh.jacobian, mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=num_nodes_per_elem,
        )
