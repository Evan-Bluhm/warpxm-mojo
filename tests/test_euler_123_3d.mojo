# 3D Einfeldt 1-2-3 problem — 1D 1-2-3 extruded along y and z.
#
# Two rarefactions move apart and leave a near-vacuum state between them
# (ρ* = 0.02185, P* = 0.001894). The run uses the Moe-Rossmanith limiter
# followed by the Zhang-Shu positivity preserver, and checks:
#   * L1 density error against the exact solution, and the centre state;
#   * mass against the exact outflow through the copy-out walls;
#   * ρ and P bounded away from zero at every node;
#   * exact uniformity in the extruded direction(s), including zero
#     transverse momentum inside each element.
# The preserver itself is unit-tested in `test_euler_positivity.mojo`.
#
# IC as in the 2D test, on [0,1] × [0,0.1]² with copy-out in x and
# periodic y, z; γ = 1.4, t_final = 0.15.

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.adjusters.positivity_euler import PositivityEuler3D
from warpxm.apps.euler import Euler
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 2
comptime NZ: Int = 2
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.15
comptime DT: Real = 1.0e-4


def _exact_rho(x: Real, t: Real) -> Real:
    """Exact density of the 1-2-3 problem: two symmetric rarefactions
    separated by a uniform near-vacuum state (Toro, test 2).

    With c0 = sqrt(γ p0 / ρ0) and ξ = |x − 1/2| / t, the fan spans
    c* ≤ ξ ≤ u0 + c0 where c* = c0 − (γ − 1) u0 / 2. Inside it
    u = 2 (ξ − c0 + (γ − 1) u0 / 2) / (γ + 1), c = ξ − u and
    ρ = ρ0 (c / c0)^(2 / (γ − 1)). For |ξ| < c*: ρ* = 0.02185, p* = 0.001894.
    """
    var u0: Real = 2.0
    var c0 = sqrt(GAMMA * Real(0.4))
    var c_star = c0 - Real(0.5) * (GAMMA - Real(1.0)) * u0
    var xi = fabs(x - Real(0.5)) / t
    if xi >= u0 + c0:
        return Real(1.0)
    var c = c_star
    if xi > c_star:
        var u = Real(2.0) * (xi - c0 + Real(0.5) * (GAMMA - Real(1.0)) * u0) / (GAMMA + Real(1.0))
        c = xi - u
    return (c / c0) ** (Real(2.0) / (GAMMA - Real(1.0)))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1), Real(0.0), Real(0.1),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=True, periodic_z=True,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState3D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var rho: Real = 1.0
                            var u: Real
                            var pp: Real = 0.4
                            if x < Real(0.5):
                                u = -2.0
                            else:
                                u = +2.0
                            var ke = Real(0.5) * rho * u * u
                            var E_tot = pp / (GAMMA - Real(1.0)) + ke
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = rho * u
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = E_tot
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    # The limiter tolerance is α = alpha_coeff · dx^1.5. The default
    # coefficient (1000) gives α ≈ 0.35, sixteen times the vacuum density,
    # so the limiter never acts there: density oscillations reach zero, the
    # positivity preserver pins a node at ρ_eps, and the sound speed at that
    # node (√(γP/ρ) ~ 10³) breaks the fixed time step. α ≈ 3.5e-3 keeps the
    # limiter active at the scale of the vacuum state.
    var limiter = MoeRossmanith3D[P, EULER_NC](
        ctx, mesh.num_elements, alpha_coeff=Real(10.0),
    )
    var positivity = PositivityEuler3D[P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)


    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState3D[P, EULER_NC]) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    # --- diagnostics over every node of every element ---------------------
    var w = NodalBasis1D[P].weights()
    var any_nan = False
    var min_rho: Real = 1.0e30
    var min_p: Real = 1.0e30
    var max_mt: Real = 0.0       # transverse momentum, exactly 0 in the IC
    var max_drift: Real = 0.0    # departure from the first transverse layer
    var mass: Real = 0.0         # per unit transverse area
    var l1_rho: Real = 0.0       # per unit transverse area
    var rho_c: Real = 0.0
    var p_c: Real = 0.0
    var u_c: Real = 0.0
    var area = ((mesh.y_max - mesh.y_min) * (mesh.z_max - mesh.z_min))
    with q.buf.map_to_host() as h:
        for et in range(NY * NZ):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                for n in range(nodes_per_elem):
                    var b = ((et * NX + i) * nodes_per_elem + n) * EULER_NC
                    var b0 = (i * nodes_per_elem + n) * EULER_NC
                    var rho = h[b + IDX_RHO]
                    var mx = h[b + IDX_MX]
                    var my = h[b + IDX_MY]
                    var mz = h[b + IDX_MZ]
                    var pp = (GAMMA - Real(1.0)) * (
                        h[b + IDX_E] - Real(0.5) * (mx * mx + my * my + mz * mz) / rho
                    )
                    if isnan(rho) or isnan(pp): any_nan = True
                    if rho < min_rho: min_rho = rho
                    if pp < min_p: min_p = pp
                    if fabs(my) > max_mt: max_mt = fabs(my)
                    if fabs(mz) > max_mt: max_mt = fabs(mz)
                    comptime for c in range(EULER_NC):
                        var d = fabs(h[b + c] - h[b0 + c])
                        if d > max_drift: max_drift = d
                    var x = xl + (Real(1.0) + ref_nodes[n % P1]) * mesh.jacobian_x
                    var jw = w[n % P1] * w[(n // P1) % P1] * w[n // (P1 * P1)] * mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z / area
                    mass += jw * rho
                    l1_rho += jw * fabs(rho - _exact_rho(x, t))
        # Centre state: mean over the two elements adjacent to x = 1/2.
        for i in range(NX // 2 - 1, NX // 2 + 1):
            for n in range(nodes_per_elem):
                var b = (i * nodes_per_elem + n) * EULER_NC
                var rho = h[b + IDX_RHO]
                var mx = h[b + IDX_MX]
                var wgt = Real(0.5) * w[n % P1] * w[(n // P1) % P1] * w[n // (P1 * P1)] / Real(8.0)
                rho_c += wgt * rho
                u_c += wgt * mx / rho
                p_c += wgt * (GAMMA - Real(1.0)) * (
                    h[b + IDX_E] - Real(0.5) * mx * mx / rho
                )

    # Until the rarefaction heads reach the walls (t < 0.5 / (2 + c0) = 0.182)
    # the copy-out boundaries see the undisturbed state, so mass leaves at
    # exactly ρ0 u0 per unit area on each side.
    var mass_exact = Real(1.0) - Real(2.0) * Real(2.0) * t
    var rho_star: Real = 0.02185

    print("3D Einfeldt 1-2-3 problem — t =", t)
    print("  min ρ, min P (all nodes):", min_rho, min_p)
    print("  max transverse momentum: ", max_mt)
    print("  max transverse drift:    ", max_drift)
    print("  mass error:              ", fabs(mass - mass_exact))
    print("  L1 ρ error vs exact:     ", l1_rho)
    print("  centre ρ, u, P:          ", rho_c, u_c, p_c, " (exact 0.02185, 0, 0.001894)")

    assert_true(not any_nan, "3D 1-2-3 NaN'd")
    assert_true(min_rho > Real(0.5) * rho_star, "density undershoots the vacuum state")
    assert_true(min_p > Real(1.0e-3), "pressure undershoots the vacuum state")
    assert_true(max_mt < Real(1.0e-12), "transverse momentum grew")
    assert_true(max_drift < Real(1.0e-12), "solution not uniform in the extruded directions")
    assert_true(fabs(mass - mass_exact) < Real(1.0e-12), "mass not conserved")
    assert_true(l1_rho < Real(1.5e-2), "L1 density error vs exact solution too large")
    assert_true(fabs(rho_c - rho_star) < Real(0.2) * rho_star, "centre density off")
    assert_true(fabs(u_c) < Real(3.0e-2), "centre velocity should vanish")
    # The centre pressure is ~4x the exact value at this resolution: the
    # well-known overheating of conservative schemes on this problem. Bound
    # it so a regression is still caught.
    assert_true(p_c < Real(1.2e-2), "centre pressure overheating got worse")
    print("PASSED")
