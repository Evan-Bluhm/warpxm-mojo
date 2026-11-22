# Nodal DG spatial solver for MixedGLM ideal MHD (9 components), generic
# over the dimension. A thin wrapper over `_dg_compute_rhs_core`. The
# hyperbolic ψ coupling is part of the flux (the extra `c_h` scalar); the
# parabolic damping of ψ is a separate source (`apply_glm_damping`).
# `USE_HLLD` selects HLLD for the MHD block instead of Rusanov.

from std.collections import Array
from max.gpu.host import DeviceContext

from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC
from warpxm.core.types import Real
from warpxm.mesh.block import BlockMesh
from warpxm.spatial.dg_core import (
    _dg_compute_rhs_core, DG_CORE_ELEMS_PER_BLOCK, InvJacobian,
)
from warpxm.spatial.dg_launch import (
    FlatLayout, IndexTensor, KernelParams, RealTensor, kernel_params,
    launch_dg,
)
from warpxm.spatial.wave_speed import cfl_dt, max_wave_speed_imhd_glm
from warpxm.variable.state import DGState


@always_inline
def _apply_bc_imhd_glm(
    q_interior: Array[Real, MHD_GLM_NC],
    bc_kind: Int,
    face_axis: Int,
) -> Array[Real, MHD_GLM_NC]:
    """Copyout ghost state for GLM-MHD (no walls supported yet)."""
    return q_interior.copy()


def _imhd_glm_rhs_kernel[NDIM: Int, P: Int, USE_HLLD: Bool](
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
    var c_h = params[3]

    def flux_x(qn: Array[Real, MHD_GLM_NC]) {imm} -> Array[Real, MHD_GLM_NC]:
        return ImhdGLM.internal_flux_x_static(qn, gamma, rho_min, press_min, c_h)

    def flux_y(qn: Array[Real, MHD_GLM_NC]) {imm} -> Array[Real, MHD_GLM_NC]:
        return ImhdGLM.internal_flux_y_static(qn, gamma, rho_min, press_min, c_h)

    def flux_z(qn: Array[Real, MHD_GLM_NC]) {imm} -> Array[Real, MHD_GLM_NC]:
        return ImhdGLM.internal_flux_z_static(qn, gamma, rho_min, press_min, c_h)

    def num_flux(
        q_l: Array[Real, MHD_GLM_NC],
        q_r: Array[Real, MHD_GLM_NC],
        nx: Real, ny: Real, nz: Real,
    ) {imm} -> Array[Real, MHD_GLM_NC]:
        comptime if USE_HLLD:
            return ImhdGLM.numerical_flux_hlld_static(
                q_l, q_r, nx, ny, nz, gamma, rho_min, press_min, c_h,
            )
        else:
            return ImhdGLM.numerical_flux_static(
                q_l, q_r, nx, ny, nz, gamma, rho_min, press_min, c_h,
            )

    def apply_bc(
        q_interior: Array[Real, MHD_GLM_NC],
        bc_kind: Int, face_axis: Int,
    ) {imm} -> Array[Real, MHD_GLM_NC]:
        return _apply_bc_imhd_glm(q_interior, bc_kind, face_axis)

    _dg_compute_rhs_core[
        NDIM, P, MHD_GLM_NC,
        FlatLayout, FlatLayout, FlatLayout, FlatLayout,
        DG_CORE_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian, Int(num_elements_i64),
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )


struct DGImhdGLMSolver[NDIM: Int, P: Int, USE_HLLD: Bool = False]:
    """DG right-hand side and CFL time step for MixedGLM MHD.

    `USE_HLLD = False` (default) uses Rusanov with c_max bumped to `c_h`;
    `USE_HLLD = True` uses IMHD HLLD for the 8 MHD components and the
    exact (B_n, ψ) 2x2 Riemann block on top.
    """

    var app: ImhdGLM

    def __init__(out self, app: ImhdGLM):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, MHD_GLM_NC],
        mut rhs: DGState[Self.NDIM, Self.P, MHD_GLM_NC],
    ) raises:
        launch_dg[_imhd_glm_rhs_kernel[Self.NDIM, Self.P, Self.USE_HLLD]](
            ctx, mesh, q, rhs,
            kernel_params(self.app.gas_gamma, self.app.min_density, self.app.min_pressure, self.app.c_h),
        )

    def suggest_dt(
        self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, MHD_GLM_NC],
    ) raises -> Real:
        """Largest stable time step for `q` at the app's `cfl_max`
        (see `spatial/wave_speed.mojo`)."""
        return cfl_dt[Self.P](
            self.app.cfl_max,
            max_wave_speed_imhd_glm(ctx, q, self.app) * mesh.inv_dx_sum(),
        )
