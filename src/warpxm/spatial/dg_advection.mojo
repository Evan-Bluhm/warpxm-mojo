# Nodal DG spatial solver for linear advection, generic over the dimension
# and the tracer count. A thin wrapper over `_dg_compute_rhs_core`: the
# kernel supplies the flux closures, the solver struct launches it.

from std.collections import Array
from max.gpu.host import DeviceContext

from warpxm.apps.advection import Advection
from warpxm.core.math_compat import fabs
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


def _advection_rhs_kernel[NDIM: Int, P: Int, NC: Int](
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
    var velocity_x = params[0]
    var velocity_y = params[1]
    var velocity_z = params[2]

    def flux_x(qn: Array[Real, NC]) {imm} -> Array[Real, NC]:
        return Advection[NC].internal_flux_x_static(qn, velocity_x)

    def flux_y(qn: Array[Real, NC]) {imm} -> Array[Real, NC]:
        return Advection[NC].internal_flux_y_static(qn, velocity_y)

    def flux_z(qn: Array[Real, NC]) {imm} -> Array[Real, NC]:
        return Advection[NC].internal_flux_z_static(qn, velocity_z)

    def num_flux(
        q_l: Array[Real, NC], q_r: Array[Real, NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, NC]:
        return Advection[NC].numerical_flux_static(
            q_l, q_r, nx, ny, nz, velocity_x, velocity_y, velocity_z,
        )

    def apply_bc(
        q_interior: Array[Real, NC], bc_kind: Int, face_axis: Int,
    ) -> Array[Real, NC]:
        # A passive scalar has no wall condition: every BC kind is copy-out.
        return q_interior.copy()

    _dg_compute_rhs_core[
        NDIM, P, NC, FlatLayout, FlatLayout, FlatLayout, FlatLayout,
        DG_CORE_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian, Int(num_elements_i64),
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGAdvectionSolver[NDIM: Int, P: Int, NC: Int]:
    """DG right-hand side for `NC` tracers advected at the app's uniform
    velocity."""

    var app: Advection[Self.NC]

    def __init__(out self, app: Advection[Self.NC]):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, Self.NC],
        mut rhs: DGState[Self.NDIM, Self.P, Self.NC],
    ) raises:
        launch_dg[_advection_rhs_kernel[Self.NDIM, Self.P, Self.NC]](
            ctx, mesh, q, rhs,
            kernel_params(
                self.app.velocity_x, self.app.velocity_y, self.app.velocity_z,
            ),
        )

    def suggest_dt(self, mesh: BlockMesh[Self.NDIM, Self.P]) -> Real:
        """Largest stable time step at the app's `cfl_max`. The velocity
        is uniform, so no state is needed."""
        var speed_over_dx = fabs(self.app.velocity_x) / mesh.dx
        comptime if Self.NDIM >= 2:
            speed_over_dx = speed_over_dx + fabs(self.app.velocity_y) / mesh.dy
        comptime if Self.NDIM >= 3:
            speed_over_dx = speed_over_dx + fabs(self.app.velocity_z) / mesh.dz
        return cfl_dt[Self.P](self.app.cfl_max, speed_over_dx)
