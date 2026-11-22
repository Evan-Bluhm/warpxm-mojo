# Nodal DG spatial solver for ideal MHD, generic over the dimension. A thin
# wrapper over `_dg_compute_rhs_core`. `USE_HLLD` selects the Miyoshi-Kusano
# HLLD flux instead of Rusanov inside the `num_flux` closure.

from std.collections import Array
from max.gpu.host import DeviceContext

from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_MX, IDX_MY, IDX_MZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.bc import BC_WALL_FREESLIP, BC_PEC_WALL
from warpxm.core.types import Real
from warpxm.mesh.block import BlockMesh
from warpxm.spatial.dg_core import (
    _dg_compute_rhs_core, DG_CORE_ELEMS_PER_BLOCK, InvJacobian,
)
from warpxm.spatial.dg_launch import (
    FlatLayout, IndexTensor, KernelParams, RealTensor, kernel_params,
    launch_dg,
)
from warpxm.spatial.wave_speed import cfl_dt, max_wave_speed_imhd
from warpxm.variable.state import DGState


@always_inline
def _apply_bc_imhd(
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


def _imhd_rhs_kernel[NDIM: Int, P: Int, USE_HLLD: Bool](
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
        return _apply_bc_imhd(q_interior, bc_kind, face_axis)

    _dg_compute_rhs_core[
        NDIM, P, MHD_NC,
        FlatLayout, FlatLayout, FlatLayout, FlatLayout,
        DG_CORE_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian, Int(num_elements_i64),
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGImhdSolver[NDIM: Int, P: Int, USE_HLLD: Bool = False]:
    """DG right-hand side and CFL time step for ideal MHD.

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
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, MHD_NC],
        mut rhs: DGState[Self.NDIM, Self.P, MHD_NC],
    ) raises:
        launch_dg[_imhd_rhs_kernel[Self.NDIM, Self.P, Self.USE_HLLD]](
            ctx, mesh, q, rhs,
            kernel_params(self.app.gas_gamma, self.app.min_density, self.app.min_pressure),
        )

    def suggest_dt(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, MHD_NC],
    ) raises -> Real:
        """Largest stable time step for `q` at the app's `cfl_max`
        (see `spatial/wave_speed.mojo`)."""
        return cfl_dt[Self.P](
            self.app.cfl_max,
            max_wave_speed_imhd(ctx, q, self.app) * mesh.inv_dx_sum(),
        )
