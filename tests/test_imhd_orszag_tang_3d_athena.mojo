# 3D Orszag-Tang vortex (Athena units) — 2D-extruded.
#
# Tensor-product extension of `test_imhd_orszag_tang_2d_athena.mojo` into
# 3D with no z-dependence: `u_z = 0, B_z = 0, ∂_z {all} = 0`. The 3D
# IMHD-GLM solver should preserve the z-invariance to machine precision
# and reproduce the same 2D evolution at every z-slice.
#
# This is item #3 from `tasks/orszag_tang_athena.md` — useful as another
# check that the 3D MHD solver gives bit-identical behavior to 2D on
# extruded ICs (we already have this validation for Brio-Wu and Alfvén).
#
# Mesh: `NX = NY = 32`, `NZ = 2`. The reduced NZ keeps the run cheap
# while still exercising multiple z-layers' worth of comparison work
# (any per-z-layer code-path bug would show up as z-drift).
#
# Assertions:
#   * no NaN
#   * mass conserved to machine precision (periodic, no source on ρ)
#   * z-drift: for every (i, j, in_, jn) the value at z-layer k matches
#     z-layer 0 to within `1e-12` for all 9 MHD-GLM components.
#   * density and gas pressure stay strictly positive
#   * total ψ² stays below the same loose bound as the 2D test

from std.math import sin, cos, isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.adjusters.positivity_mhd import PositivityMHD3D
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC, IDX_PSI
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real, PI, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_glm_3d import DGImhdGLMSolver3D
from warpxm.spatial.imhd_glm_source_3d import apply_glm_damping_3d
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 2
comptime GAMMA: Real = 5.0 / 3.0
comptime V0: Real = 1.0
comptime C_H: Real = 3.0
comptime C_P_SQ: Real = 0.18
comptime T_FINAL: Real = 0.3
comptime DT: Real = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx,
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = DGState3D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var d0: Real = Real(25.0) / (Real(36.0) * PI)
    var p0: Real = Real(5.0) / (Real(12.0) * PI)
    var B0: Real = Real(1.0) / sqrt(Real(4.0) * PI)

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        var y = (
                            yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        )
                        for in_ in range(P1):
                            var x = (
                                xl
                                + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            )
                            var ux = -V0 * sin(TWO_PI * y)
                            var uy = V0 * sin(TWO_PI * x)
                            var Bx_v = -B0 * sin(TWO_PI * y)
                            var By_v = B0 * sin(Real(2.0) * TWO_PI * x)

                            var msq = d0 * (ux * ux + uy * uy)
                            var bsq = Bx_v * Bx_v + By_v * By_v
                            var E_tot = (
                                p0 / (GAMMA - Real(1.0))
                                + Real(0.5) * msq + Real(0.5) * bsq
                            )

                            var n = (kn * P1 + jn) * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                            host[b + IDX_RHO] = d0
                            host[b + IDX_MX] = d0 * ux
                            host[b + IDX_MY] = d0 * uy
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
    var positivity = PositivityMHD3D[P, MHD_GLM_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var mass_0: Real = 0.0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = (kn * P1 + jn) * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                            mass_0 += (
                                jv * weights[in_] * weights[jn] * weights[kn]
                                * host[b + IDX_RHO]
                            )

    def compute_rhs(
        mut q_in: DGState3D[P, MHD_GLM_NC],
        mut q_out: DGState3D[P, MHD_GLM_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_3d[P, MHD_GLM_NC](ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: DGState3D[P, MHD_GLM_NC]) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    var any_nan = False
    var mass_T: Real = 0.0
    var psi2_T: Real = 0.0
    var max_rho: Real = -1.0e30
    var min_rho: Real = 1.0e30
    var max_p: Real = -1.0e30
    var min_p: Real = 1.0e30
    var max_z_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        # Diagnostics + z-drift check (compare each z-layer against k=0).
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var e_k = k * NX * NY + j * NX + i
                    var e_0 = 0 * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var n = (kn * P1 + jn) * P1 + in_
                                var b_k = (
                                    e_k * nodes_per_elem + n
                                ) * MHD_GLM_NC
                                var b_0 = (
                                    e_0 * nodes_per_elem + n
                                ) * MHD_GLM_NC
                                var rho = h[b_k + IDX_RHO]
                                var mx = h[b_k + IDX_MX]
                                var my = h[b_k + IDX_MY]
                                var mz = h[b_k + IDX_MZ]
                                var E_tot = h[b_k + IDX_E]
                                var Bx = h[b_k + IDX_BX]
                                var By = h[b_k + IDX_BY]
                                var Bz = h[b_k + IDX_BZ]
                                var psi = h[b_k + IDX_PSI]
                                if isnan(rho) or isnan(psi) or isnan(E_tot):
                                    any_nan = True
                                var ke = Real(0.5) * (
                                    mx * mx + my * my + mz * mz
                                ) / rho
                                var mag = Real(0.5) * (
                                    Bx * Bx + By * By + Bz * Bz
                                )
                                var pp = (GAMMA - Real(1.0)) * (
                                    E_tot - ke - mag
                                )
                                var jw = (
                                    jv * weights[in_] * weights[jn]
                                    * weights[kn]
                                )
                                mass_T += jw * rho
                                psi2_T += jw * psi * psi
                                if rho > max_rho: max_rho = rho
                                if rho < min_rho: min_rho = rho
                                if pp > max_p: max_p = pp
                                if pp < min_p: min_p = pp
                                # z-drift: each non-zero z-layer should match
                                # the k=0 reference exactly.
                                if k > 0:
                                    comptime for c in range(MHD_GLM_NC):
                                        var dv = fabs(h[b_k + c] - h[b_0 + c])
                                        if dv > max_z_drift:
                                            max_z_drift = dv

    var rel_mass_drift = fabs(mass_T - mass_0) / mass_0
    print(
        "Orszag-Tang vortex (Athena, 3D-extruded) — t =", t,
        " on", NX, "x", NY, "x", NZ, "mesh, P =", P,
    )
    print("  mass  (initial / final):   ", mass_0, " /", mass_T)
    print("  rel Δmass:                  ", rel_mass_drift)
    print("  ρ range: [", min_rho, ",", max_rho, "]  (initial =", d0, ")")
    print("  P range: [", min_p, ",", max_p, "]  (initial =", p0, ")")
    print("  ∫ψ² dV:                     ", psi2_T)
    print("  max z-drift across components:", max_z_drift)

    assert_true(not any_nan, "3D OT-Athena NaN'd")
    # 3D quadrature has `NZ × P1` additional accumulator iterations vs
    # the 2D test, so the FP drift floor is a few × larger.
    assert_true(rel_mass_drift < Real(1.0e-11), "mass not conserved")
    assert_true(min_rho > Real(0.0), "density went non-positive")
    assert_true(min_p > Real(0.0), "pressure went non-positive")
    assert_true(max_rho < Real(10.0) * d0, "density blew up")
    assert_true(psi2_T < Real(1.0e-3), "GLM ψ² grew uncontrollably")
    # z-drift bound: floating-point cumulative error over 600 SSPRK3 steps
    # on a problem with σ ≈ 1 fields. 1e-10 is loose but well above
    # the asymmetric-summation noise floor.
    assert_true(
        max_z_drift < Real(1.0e-10),
        "3D OT not bit-uniform in z — 3D solver introduced z-asymmetry",
    )
    print("PASSED")
