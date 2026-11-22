# Nodal DG spatial solver for 5-moment Euler, generic over the dimension.
# A thin wrapper over `_dg_compute_rhs_core`: the kernel unpacks the Euler
# scalars (γ, ρ_min, P_min) and supplies the flux and BC closures; the solver
# struct launches it and provides the CFL time step.

from std.collections import Array
from max.gpu.host import DeviceContext

from warpxm.apps.euler import Euler, EULER_NC, IDX_MX, IDX_MY, IDX_MZ
from warpxm.core.bc import BC_WALL_FREESLIP
from warpxm.core.types import Real
from warpxm.mesh.block import BlockMesh
from warpxm.spatial.dg_core import (
    _dg_compute_rhs_core, DG_CORE_ELEMS_PER_BLOCK, InvJacobian,
)
from warpxm.spatial.dg_launch import (
    FlatLayout, IndexTensor, KernelParams, RealTensor, kernel_params,
    launch_dg,
)
from warpxm.spatial.wave_speed import cfl_dt, max_wave_speed_euler
from warpxm.variable.state import DGState


@always_inline
def _apply_bc_euler(
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


def _euler_rhs_kernel[NDIM: Int, P: Int](
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
    var gamma = params[0]
    var rho_min = params[1]
    var press_min = params[2]

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
        return _apply_bc_euler(q_interior, bc_kind, face_axis)

    _dg_compute_rhs_core[
        NDIM, P, EULER_NC,
        FlatLayout, FlatLayout, FlatLayout, FlatLayout,
        DG_CORE_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian, Int(num_elements_i64),
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGEulerSolver[NDIM: Int, P: Int]:
    """DG right-hand side and CFL time step for 5-moment Euler."""

    var app: Euler

    def __init__(out self, app: Euler):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, EULER_NC],
        mut rhs: DGState[Self.NDIM, Self.P, EULER_NC],
    ) raises:
        launch_dg[_euler_rhs_kernel[Self.NDIM, Self.P]](
            ctx, mesh, q, rhs,
            kernel_params(self.app.gas_gamma, self.app.min_density, self.app.min_pressure),
        )

    def suggest_dt(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, EULER_NC],
    ) raises -> Real:
        """Largest stable time step for `q` at the app's `cfl_max`
        (see `spatial/wave_speed.mojo`)."""
        return cfl_dt[Self.P](
            self.app.cfl_max,
            max_wave_speed_euler(ctx, q, self.app) * mesh.inv_dx_sum(),
        )
