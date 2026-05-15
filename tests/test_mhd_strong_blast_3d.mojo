# 3D strong MHD blast — 3D analog of `test_mhd_strong_blast_2d.mojo`.
# Spherical pressure pulse (ΔP/P_amb = 100, σ = 0.05) in a strong B₀ = 5
# magnetic background. Without positivity preservation the simulation
# NaNs as the shock fronts drive P_gas through zero.

from std.math import exp, isnan
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith_3d import MoeRossmanith3D
from warpxm.adjusters.positivity_mhd_3d import PositivityMHD3D
from warpxm.apps.imhd_3d import IMHD3D
from warpxm.apps.imhd_2d import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.05
comptime P_AMB: Real = 0.1
comptime DELTA_P: Real = 100.0
comptime B0: Real = 5.0
comptime T_FINAL: Real = 0.02
comptime DT: Real = 2.5e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)

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
                            var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
                            var bsq = B0 * B0
                            var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_NC
                            host[b + IDX_RHO] = Real(1.0)
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = E_tot
                            host[b + IDX_BX] = B0
                            host[b + IDX_BY] = Real(0.0)
                            host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var mass_0: Real = 0.0
    var bx_0: Real = 0.0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_NC
                            var jw = jv * weights[in_] * weights[jn] * weights[kn]
                            mass_0 += jw * host[b + IDX_RHO]
                            bx_0 += jw * host[b + IDX_BX]

    var app = IMHD3D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver3D[P](app)
    var stepper = SSPRK3_3D[P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, MHD_NC](ctx, mesh.num_elements)
    var positivity = PositivityMHD3D[P, MHD_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )


    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, MHD_NC],
        mut q_out: DGState3D[P, MHD_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState3D[P, MHD_NC]) raises:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var any_nan = False
    var min_rho: Real = 1.0e30
    var min_p: Real = 1.0e30
    var mass_T: Real = 0.0
    var bx_T: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MHD_NC
                                var rho = h[b + IDX_RHO]
                                var mx = h[b + IDX_MX]
                                var my = h[b + IDX_MY]
                                var mz = h[b + IDX_MZ]
                                var E = h[b + IDX_E]
                                var Bx = h[b + IDX_BX]
                                var By = h[b + IDX_BY]
                                var Bz = h[b + IDX_BZ]
                                if isnan(rho) or isnan(E): any_nan = True
                                var ke = Real(0.5) * (mx * mx + my * my + mz * mz) / rho
                                var mag = Real(0.5) * (Bx * Bx + By * By + Bz * Bz)
                                var pp = (GAMMA - Real(1.0)) * (E - ke - mag)
                                if rho < min_rho: min_rho = rho
                                if pp < min_p: min_p = pp
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                mass_T += jw * rho
                                bx_T += jw * Bx

    print("3D strong MHD blast — t =", t)
    print("  min ρ:                    ", min_rho)
    print("  min P_gas:                ", min_p)
    print("  rel Δmass:                ", fabs(mass_T - mass_0) / mass_0)
    print("  rel Δ(∫B_x dV):           ", fabs(bx_T - bx_0) / bx_0)

    assert_true(not any_nan, "3D strong MHD blast NaN'd")
    assert_true(min_rho > Real(0.0), "negative ρ survived preserver")
    assert_true(min_p > Real(0.0), "negative P_gas survived preserver")
    assert_true(fabs(bx_T - bx_0) / bx_0 < Real(1.0e-10),
                "∫B_x not preserved")
    print("PASSED")
