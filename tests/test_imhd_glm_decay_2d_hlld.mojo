# HLLD path of the 2D GLM-MHD ψ-decay regression.
#
# Same IC as test_imhd_glm_decay_2d.mojo, but the spatial solver is
# DGImhdGLMSolver2D[P, USE_HLLD=True]. Validates that the new HLLD code path
# preserves the same qualitative GLM behaviour: ψ charges from the ∇·B
# coupling, then decays under the parabolic damping; B_x² stays bounded.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd_glm import (
    ImhdGLM, MHD_GLM_NC, IDX_PSI,
)
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_glm_2d import DGImhdGLMSolver2D
from warpxm.spatial.imhd_glm_source_2d import apply_glm_damping_2d
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime DELTA: Real = 0.1
comptime SIGMA: Real = 0.1
comptime C_H: Real = 1.0
comptime C_P_SQ: Real = 0.05
comptime DT: Real = 5.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var r2 = (x - Real(0.5)) * (x - Real(0.5)) + (y - Real(0.5)) * (y - Real(0.5))
                    var Bx_bump = DELTA * exp(-r2 / (SIGMA * SIGMA))
                    var rho: Real = 1.0
                    var pp: Real = 1.0
                    var bsq = Bx_bump * Bx_bump
                    var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
                    host[b + IDX_BX] = Bx_bump
                    host[b + IDX_BY] = Real(0.0)
                    host[b + IDX_BZ] = Real(0.0)
                    host[b + IDX_PSI] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver2D[P, USE_HLLD=True](app)
    var stepper = SSPRK3_2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var ja = mesh.jacobian_x * mesh.jacobian_y

    var psi2_0: Real = 0.0
    var bx2_0: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                        var jw = ja * weights[in_] * weights[jn]
                        psi2_0 += jw * h[b + IDX_PSI] * h[b + IDX_PSI]
                        bx2_0 += jw * h[b + IDX_BX] * h[b + IDX_BX]
    print("HLLD t = 0     : ∫ψ²  =", psi2_0, "  ∫B_x² =", bx2_0)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MHD_GLM_NC],
        mut q_out: DGState2D[P, MHD_GLM_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_2d[P](ctx, q_in, q_out, C_H, C_P_SQ)

    @parameter
    def no_adjust(mut q_: DGState2D[P, MHD_GLM_NC]) raises:
        pass

    var t: Real = 0.0
    var t_charge: Real = 0.5
    var t_final: Real = 4.0
    while t < t_charge - Real(0.5) * DT:
        stepper.step[compute_rhs, no_adjust](ctx, q, DT)
        t += DT
    ctx.synchronize()
    var psi2_max: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                        var jw = ja * weights[in_] * weights[jn]
                        psi2_max += jw * h[b + IDX_PSI] * h[b + IDX_PSI]
    print("HLLD t =", t, ": ∫ψ²  =", psi2_max, "  (charging done)")

    while t < t_final - Real(0.5) * DT:
        stepper.step[compute_rhs, no_adjust](ctx, q, DT)
        t += DT
    ctx.synchronize()

    var psi2_end: Real = 0.0
    var bx2_end: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                        var jw = ja * weights[in_] * weights[jn]
                        psi2_end += jw * h[b + IDX_PSI] * h[b + IDX_PSI]
                        bx2_end += jw * h[b + IDX_BX] * h[b + IDX_BX]
    print("HLLD t =", t, ": ∫ψ²  =", psi2_end, "  ∫B_x² =", bx2_end)
    print("HLLD ψ² decay ratio (end / max):", psi2_end / psi2_max)

    assert_true(psi2_max > psi2_0 + Real(1.0e-8),
                "ψ should pick up energy from ∇·B coupling (HLLD)")
    assert_true(psi2_end < Real(0.5) * psi2_max,
                "ψ should decay at least 50 % from its peak under GLM damping (HLLD)")
    assert_true(bx2_end < Real(2.0) * bx2_0,
                "B_x² should stay bounded — no runaway growth (HLLD)")
    print("PASSED")
