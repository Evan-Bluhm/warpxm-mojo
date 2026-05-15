# HLLD path of the 3D GLM-MHD ψ-decay regression.
#
# Same IC as test_imhd_glm_decay_3d.mojo, but the spatial solver is
# DGImhdGLMSolver3D[P, USE_HLLD=True]. Confirms the new 3D HLLD GLM code
# path stays bounded, charges ψ from ∇·B, and decays ψ under the parabolic
# damping.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd_glm_3d import ImhdGLM3D, MHD_GLM_NC_3D, IDX_PSI_3D
from warpxm.apps.imhd_2d import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_glm_3d import DGImhdGLMSolver3D
from warpxm.spatial.imhd_glm_source_3d import apply_glm_damping_3d
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime DELTA: Real = 0.1
comptime SIGMA: Real = 0.1
comptime C_H: Real = 1.0
comptime C_P_SQ: Real = 0.05
comptime DT: Real = 5.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, MHD_GLM_NC_3D](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        var zl = mesh.z_min + Real(k) * mesh.dz
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var r2 = (x - Real(0.5)) * (x - Real(0.5))
                            r2 += (y - Real(0.5)) * (y - Real(0.5))
                            r2 += (z - Real(0.5)) * (z - Real(0.5))
                            var Bx_bump = DELTA * exp(-r2 / (SIGMA * SIGMA))
                            var rho: Real = 1.0
                            var pp: Real = 1.0
                            var bsq = Bx_bump * Bx_bump
                            var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_GLM_NC_3D
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = E_tot
                            host[b + IDX_BX] = Bx_bump
                            host[b + IDX_BY] = Real(0.0)
                            host[b + IDX_BZ] = Real(0.0)
                            host[b + IDX_PSI_3D] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = ImhdGLM3D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver3D[P, USE_HLLD=True](app)
    var stepper = SSPRK3_3D[P, MHD_GLM_NC_3D](ctx, mesh.num_elements)

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z

    var psi2_max: Real = 0.0

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, MHD_GLM_NC_3D],
        mut q_out: DGState3D[P, MHD_GLM_NC_3D],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_3d[P, MHD_GLM_NC_3D](ctx, q_in, q_out, C_H, C_P_SQ)

    @parameter
    def no_adjust(mut q_: DGState3D[P, MHD_GLM_NC_3D]) raises:
        pass

    var t: Real = 0.0
    var t_charge: Real = 0.5
    var t_final: Real = 4.0
    while t < t_charge - Real(0.5) * DT:
        stepper.step[compute_rhs, no_adjust](ctx, q, DT)
        t += DT
    ctx.synchronize()
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MHD_GLM_NC_3D
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                psi2_max += jw * h[b + IDX_PSI_3D] * h[b + IDX_PSI_3D]
    print("HLLD t =", t, ": ∫ψ²  =", psi2_max, "  (charging done)")

    while t < t_final - Real(0.5) * DT:
        stepper.step[compute_rhs, no_adjust](ctx, q, DT)
        t += DT
    ctx.synchronize()

    var psi2_end: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MHD_GLM_NC_3D
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                psi2_end += jw * h[b + IDX_PSI_3D] * h[b + IDX_PSI_3D]

    print("HLLD t =", t, ": ∫ψ²  =", psi2_end)
    print("HLLD ψ² decay ratio (end / charging peak):", psi2_end / psi2_max)
    assert_true(psi2_max > Real(1.0e-10), "ψ should pick up energy from ∇·B coupling (HLLD)")
    assert_true(psi2_end < Real(0.5) * psi2_max, "ψ should decay ≥ 50 % from peak (HLLD)")
    print("PASSED")
