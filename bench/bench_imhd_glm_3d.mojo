# 3D IMHD-GLM bench driver.

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC, IDX_PSI
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_glm_3d import DGImhdGLMSolver3D
from warpxm.spatial.imhd_glm_source_3d import apply_glm_damping_3d
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime C_H: Real = 4.0
comptime C_P_SQ: Real = 0.5
comptime DT: Real = 5.0e-4
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx,
        Real(0.0), TWO_PI,
        Real(0.0), TWO_PI,
        Real(0.0), TWO_PI,
        nx=NX, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = DGState3D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1 * P1
    var rho_0 = GAMMA * GAMMA
    var p_0: Real = GAMMA
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    # z-coord not used by the 2D-extruded OT-like IC.
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = (
                                xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            )
                            var ux = -sin(y)
                            var uy = sin(x)
                            var Bx_v = -sin(y)
                            var By_v = sin(Real(2.0) * x)
                            var rho = rho_0
                            var msq = rho * (ux * ux + uy * uy)
                            var bsq = Bx_v * Bx_v + By_v * By_v
                            var E_tot = (
                                p_0 / (GAMMA - Real(1.0))
                                + Real(0.5) * msq + Real(0.5) * bsq
                            )
                            var n = (kn * P1 + jn) * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = rho * ux
                            host[b + IDX_MY] = rho * uy
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = E_tot
                            host[b + IDX_BX] = Bx_v
                            host[b + IDX_BY] = By_v
                            host[b + IDX_BZ] = Real(0.0)
                            host[b + IDX_PSI] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver3D[P](app)
    var stepper = SSPRK3_3D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, MHD_GLM_NC],
        mut q_out: DGState3D[P, MHD_GLM_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_3d[P, MHD_GLM_NC](ctx, q_in, q_out, C_H, C_P_SQ)

    @parameter
    def adjust(mut q_: DGState3D[P, MHD_GLM_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    for _ in range(WARMUP):
        stepper.step[compute_rhs, adjust](ctx, q, DT)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step[compute_rhs, adjust](ctx, q, DT)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var elapsed_s = Real(t1 - t0) / Real(1.0e9)
    var steps_per_s = Real(N_STEPS) / elapsed_s
    print(
        "BENCH config=imhd_glm_3d steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
    )
