# 3D Euler blast wave on a triply-periodic mesh.
#
# IC:
#     ρ(x, y, z, 0) = 1
#     u = 0
#     P(x, y, z, 0) = P_amb + ΔP · exp(−r² / σ²)
#     E = P / (γ − 1)
#
# with `r = |x − x_c|` and the perturbation centred at the domain centre.
# The pressure pulse drives an outgoing spherical shock; the negative-density
# / negative-pressure failure mode the limiter is designed to suppress
# triggers immediately for ΔP/P_amb ≳ 5 without it.
#
# We check at moderate t = 0.1 (well past shock formation):
#   * no NaN
#   * mass conservation to machine precision (periodic, no sources)
#   * density stays positive
#   * fields remain bounded

from std.math import exp, isnan
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.euler import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler import Euler
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime DELTA_P: Real = 5.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, EULER_NC](ctx, mesh.num_elements)

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
                            var rho: Real = 1.0
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    # Compute initial mass.
    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var mass_0: Real = 0.0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            var jw = jv * weights[in_] * weights[jn] * weights[kn]
                            mass_0 += jw * host[b + IDX_RHO]

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, EULER_NC](ctx, mesh.num_elements)


    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState3D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var any_nan = False
    var mass_T: Real = 0.0
    var min_rho: Real = 1.0e30
    var max_rho: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * EULER_NC
                                var rho = h[b + IDX_RHO]
                                var E_tot = h[b + IDX_E]
                                if isnan(rho) or isnan(E_tot):
                                    any_nan = True
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                mass_T += jw * rho
                                if rho < min_rho: min_rho = rho
                                if rho > max_rho: max_rho = rho

    var rel_mass_drift = fabs(mass_T - mass_0) / mass_0
    print("3D blast wave — t =", t)
    print("  initial mass:  ", mass_0)
    print("  final mass:    ", mass_T)
    print("  rel Δmass:      ", rel_mass_drift)
    print("  ρ range: [", min_rho, ",", max_rho, "]")
    assert_true(not any_nan, "3D blast wave NaN'd")
    assert_true(rel_mass_drift < Real(1.0e-12), "mass not conserved on periodic mesh")
    assert_true(min_rho > Real(0.0), "density went non-positive")
    assert_true(max_rho < Real(10.0), "density blew up — limiter unstable")
    print("PASSED")
