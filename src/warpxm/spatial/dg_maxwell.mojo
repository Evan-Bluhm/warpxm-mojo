# Nodal DG spatial solver for Maxwell's equations, generic over the
# dimension. A thin wrapper over `_dg_compute_rhs_core`. The only physics
# scalar is `c_light`; the BC function supports PEC and PMC walls.
#
# All six field components are evolved in every dimension. In 1D, waves
# propagate along x: (E_y, B_z) and (E_z, B_y) couple through the x-flux
# and E_x, B_x are stationary.

from std.collections import Array
from max.gpu.host import DeviceContext

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.bc import BC_PEC_WALL, BC_PMC_WALL
from warpxm.core.types import Real
from warpxm.mesh.block import BlockMesh
from warpxm.spatial.dg_core import (
    _dg_compute_rhs_core, DG_CORE_ELEMS_PER_BLOCK, InvJacobian,
)
from warpxm.spatial.dg_launch import (
    FlatLayout, IndexTensor, KernelParams, RealTensor, kernel_params,
    launch_dg,
)
from warpxm.spatial.wave_speed import cfl_dt
from warpxm.variable.state import DGState


@always_inline
def _apply_bc_maxwell(
    q_interior: Array[Real, MAXWELL_NC],
    bc_kind: Int,
    face_axis: Int,
) -> Array[Real, MAXWELL_NC]:
    """Ghost-state synthesis. `face_axis` ∈ {0, 1, 2} selects the wall normal.

    `BC_PEC_WALL`: flip the two tangential E components and the normal B.
    `BC_PMC_WALL`: dual — flip the normal E and the two tangential B.
    Anything else copies the interior state.
    """
    var q: Array[Real, MAXWELL_NC] = q_interior.copy()
    if bc_kind == BC_PEC_WALL:
        if face_axis == 0:
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BX] = -q_interior[IDX_BX]
        elif face_axis == 1:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BY] = -q_interior[IDX_BY]
        else:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_BZ] = -q_interior[IDX_BZ]
    elif bc_kind == BC_PMC_WALL:
        if face_axis == 0:
            q[IDX_EX] = -q_interior[IDX_EX]
            q[IDX_BY] = -q_interior[IDX_BY]
            q[IDX_BZ] = -q_interior[IDX_BZ]
        elif face_axis == 1:
            q[IDX_EY] = -q_interior[IDX_EY]
            q[IDX_BX] = -q_interior[IDX_BX]
            q[IDX_BZ] = -q_interior[IDX_BZ]
        else:
            q[IDX_EZ] = -q_interior[IDX_EZ]
            q[IDX_BX] = -q_interior[IDX_BX]
            q[IDX_BY] = -q_interior[IDX_BY]
    return q^


def _maxwell_rhs_kernel[NDIM: Int, P: Int](
    q: RealTensor,
    rhs: RealTensor,
    face_neg_elem: IndexTensor,
    face_pos_elem: IndexTensor,
    elem_face_dir: IndexTensor,
    face_bc: IndexTensor,
    inv_jacobian: InvJacobian,
    num_elements_i64: Int64,
    params: KernelParams,
):
    var c_light = params[0]

    def flux_x(qn: Array[Real, MAXWELL_NC]) {imm} -> Array[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_x_static(qn, c_light)

    def flux_y(qn: Array[Real, MAXWELL_NC]) {imm} -> Array[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_y_static(qn, c_light)

    def flux_z(qn: Array[Real, MAXWELL_NC]) {imm} -> Array[Real, MAXWELL_NC]:
        return Maxwell.internal_flux_z_static(qn, c_light)

    def num_flux(
        q_l: Array[Real, MAXWELL_NC],
        q_r: Array[Real, MAXWELL_NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, MAXWELL_NC]:
        return Maxwell.numerical_flux_static(q_l, q_r, nx, ny, nz, c_light)

    def apply_bc(
        q_interior: Array[Real, MAXWELL_NC],
        bc_kind: Int, face_axis: Int,
    ) {imm} -> Array[Real, MAXWELL_NC]:
        return _apply_bc_maxwell(q_interior, bc_kind, face_axis)

    _dg_compute_rhs_core[
        NDIM, P, MAXWELL_NC,
        FlatLayout, FlatLayout, FlatLayout, FlatLayout,
        DG_CORE_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian, Int(num_elements_i64),
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGMaxwellSolver[NDIM: Int, P: Int]:
    """DG right-hand side and CFL time step for Maxwell's equations."""

    var app: Maxwell

    def __init__(out self, app: Maxwell):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, MAXWELL_NC],
        mut rhs: DGState[Self.NDIM, Self.P, MAXWELL_NC],
    ) raises:
        launch_dg[_maxwell_rhs_kernel[Self.NDIM, Self.P]](
            ctx, mesh, q, rhs,
            kernel_params(self.app.c_light),
        )

    def suggest_dt(self, mesh: BlockMesh[Self.NDIM, Self.P]) -> Real:
        """Largest stable time step at the app's `cfl_max`. Light speed is
        constant, so no state is needed."""
        return cfl_dt[Self.P](
            self.app.cfl_max, self.app.c_light * mesh.inv_dx_sum(),
        )
