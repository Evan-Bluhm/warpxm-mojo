# 1D nodal DG spatial solver for advection — thin wrapper over the
# shared `_dg_1d_compute_rhs_core` template. NC-parameterised over the
# tracer count.

from std.collections import Array
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.advection import Advection
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_core_1d import (
    _dg_1d_compute_rhs_core, DG_CORE_1D_ELEMS_PER_BLOCK,
)
from warpxm.variable.state_1d import DGState1D


@always_inline
def _apply_bc_scalar[NC: Int](
    q_interior: Array[Real, NC], bc_kind: Int,
) -> Array[Real, NC]:
    """Ghost-state for a passive scalar — copyout regardless of BC kind."""
    return q_interior.copy()


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
    num_elements_i64: Int64,
):
    var num_elements = Int(num_elements_i64)
    def flux_x(qn: Array[Real, NC]) {imm} -> Array[Real, NC]:
        return Advection[NC].internal_flux_x_static(qn, velocity)

    def num_flux(
        q_l: Array[Real, NC], q_r: Array[Real, NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, NC]:
        return Advection[NC].numerical_flux_static(
            q_l, q_r, nx, ny, nz, velocity, Real(0.0), Real(0.0),
        )

    def apply_bc(
        q_interior: Array[Real, NC], bc_kind: Int,
    ) {} -> Array[Real, NC]:
        return _apply_bc_scalar[NC](q_interior, bc_kind)

    _dg_1d_compute_rhs_core[
        P, NC, QLayout, RhsLayout, ElemFaceLayout,
        DG_CORE_1D_ELEMS_PER_BLOCK,
    ](
        q, rhs, elem_left_face, elem_right_face,
        face_left, face_right, face_bc,
        jacobian, num_elements,
        flux_x, num_flux, apply_bc,
    )


struct DGSpatialSolver1D[P: Int, NC: Int]:
    """Host-side launcher for the multi-component 1D advection DG kernel."""

    var app: Advection[Self.NC]

    def __init__(out self, app: Advection[Self.NC]):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, Self.NC],
        mut rhs: DGState1D[Self.P, Self.NC],
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

        comptime kernel = _advection_rhs_kernel[
            Self.P, Self.NC,
            type_of(q_layout), type_of(q_layout), type_of(elem_face_layout),
        ]
        comptime EB = DG_CORE_1D_ELEMS_PER_BLOCK
        ctx.enqueue_function[kernel](
            q_view, rhs_view, elf_view, erf_view, fl_view, fr_view, fbc_view,
            self.app.velocity_x, mesh.jacobian, Int64(mesh.num_elements),
            grid_dim=ceildiv(mesh.num_elements, EB),
            block_dim=(num_nodes_per_elem, EB),
        )
