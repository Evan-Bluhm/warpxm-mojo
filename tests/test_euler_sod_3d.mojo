# 3D Sod shock tube — 1D physics extruded along y and z.
#
# Initial discontinuity at x = 0.5; uniform in y and z. Sides perpendicular
# to x are copyout outflows; top/bottom and down/up are free-slip walls.
# Validates the 3D non-periodic BC plumbing on a problem with a known
# analytic structure.
#
# Expected outcome at t = 0.2:
#   * No NaN
#   * Result uniform in y and z at every x
#   * x-profile matches the 1D Sod analytic solution (within shock-capturing
#     tolerance) at four sample locations, and in L1 for (ρ, u, P) against
#     the exact Riemann solution
#   * Mass, x-momentum and energy obey the exact boundary-flux balance

from std.collections import Array
from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.euler import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler import Euler
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 100
comptime NY: Int = 2
comptime NZ: Int = 2
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime DT: Real = 5.0e-4


# ---- Exact Riemann solution (Toro, "Riemann Solvers and Numerical Methods
# for Fluid Dynamics", 3rd ed., §4.2–4.5) --------------------------------

def _wave_fn(p: Real, rho_k: Real, p_k: Real) -> Real:
    """Toro's f_K(p): velocity change across the K-wave for star pressure p
    (shock branch if p > p_k, rarefaction branch otherwise)."""
    if p > p_k:
        var a = Real(2.0) / ((GAMMA + Real(1.0)) * rho_k)
        var b = (GAMMA - Real(1.0)) / (GAMMA + Real(1.0)) * p_k
        return (p - p_k) * sqrt(a / (p + b))
    var c_k = sqrt(GAMMA * p_k / rho_k)
    var ex = (GAMMA - Real(1.0)) / (Real(2.0) * GAMMA)
    return Real(2.0) * c_k / (GAMMA - Real(1.0)) * ((p / p_k) ** ex - Real(1.0))


def _star_pressure(
    rho_l: Real, u_l: Real, p_l: Real, rho_r: Real, u_r: Real, p_r: Real,
) -> Real:
    """Solve f_L(p) + f_R(p) + u_R − u_L = 0 by bisection (f is monotone)."""
    var lo: Real = 0.0
    var hi: Real = Real(10.0) * (p_l + p_r)
    for _ in range(100):
        var mid = Real(0.5) * (lo + hi)
        var f = _wave_fn(mid, rho_l, p_l) + _wave_fn(mid, rho_r, p_r) + u_r - u_l
        if f > Real(0.0):
            hi = mid
        else:
            lo = mid
    return Real(0.5) * (lo + hi)


def _sample_left(
    rho_k: Real, u_k: Real, p_k: Real, p_s: Real, u_s: Real, xi: Real,
) -> Array[Real, 3]:
    """(ρ, u, p) at similarity coordinate xi = (x − x₀)/t on the left of the
    contact (xi < u*). The right side is obtained by mirror symmetry."""
    var out = Array[Real, 3](fill=Real(0.0))
    var c_k = sqrt(GAMMA * p_k / rho_k)
    var gm = (GAMMA - Real(1.0)) / (GAMMA + Real(1.0))
    out[0] = rho_k; out[1] = u_k; out[2] = p_k
    if p_s > p_k:
        # Left shock.
        var s = u_k - c_k * sqrt(
            (GAMMA + Real(1.0)) / (Real(2.0) * GAMMA) * p_s / p_k
            + (GAMMA - Real(1.0)) / (Real(2.0) * GAMMA)
        )
        if xi > s:
            out[0] = rho_k * (p_s / p_k + gm) / (gm * p_s / p_k + Real(1.0))
            out[1] = u_s; out[2] = p_s
    else:
        # Left rarefaction: head, fan, tail.
        var c_s = c_k * (p_s / p_k) ** ((GAMMA - Real(1.0)) / (Real(2.0) * GAMMA))
        if xi > u_s - c_s:
            out[0] = rho_k * (p_s / p_k) ** (Real(1.0) / GAMMA)
            out[1] = u_s; out[2] = p_s
        elif xi > u_k - c_k:
            var c = Real(2.0) / (GAMMA + Real(1.0)) * (
                c_k + Real(0.5) * (GAMMA - Real(1.0)) * (u_k - xi)
            )
            out[0] = rho_k * (c / c_k) ** (Real(2.0) / (GAMMA - Real(1.0)))
            out[1] = Real(2.0) / (GAMMA + Real(1.0)) * (
                c_k + Real(0.5) * (GAMMA - Real(1.0)) * u_k + xi
            )
            out[2] = p_k * (c / c_k) ** (Real(2.0) * GAMMA / (GAMMA - Real(1.0)))
    return out^


def _exact_riemann(
    rho_l: Real, u_l: Real, p_l: Real, rho_r: Real, u_r: Real, p_r: Real,
    p_s: Real, xi: Real,
) -> Array[Real, 3]:
    var u_s = Real(0.5) * (u_l + u_r) + Real(0.5) * (
        _wave_fn(p_s, rho_r, p_r) - _wave_fn(p_s, rho_l, p_l)
    )
    if xi < u_s:
        return _sample_left(rho_l, u_l, p_l, p_s, u_s, xi)
    var m = _sample_left(rho_r, -u_r, p_r, p_s, -u_s, -xi)
    m[1] = -m[1]
    return m^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1), Real(0.0), Real(0.1),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=False, periodic_z=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
        bc_down=BC_WALL_FREESLIP, bc_up=BC_WALL_FREESLIP,
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
                            var rho: Real
                            var pp: Real
                            if x < Real(0.5):
                                rho = 1.0; pp = 1.0
                            else:
                                rho = 0.125; pp = 0.1
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, EULER_NC](ctx, mesh.num_elements)


    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState3D[P, EULER_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    # Pull the mid-row (j=0, k=0) midpoint sample at every x.
    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var max_y_drift: Real = 0.0
    var max_z_drift: Real = 0.0
    var rho_row = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        # First pass: row 0 in z=0 plane.
        for i in range(NX):
            var e = 0 * NX * NY + 0 * NX + i
            var b = (e * nodes_per_elem + mid_node) * EULER_NC
            var rho_ji = h[b + IDX_RHO]
            if isnan(rho_ji): any_nan = True
            rho_row[i] = rho_ji
        # Compare every other (j, k) sample to the row-0 z-0 value at the same x.
        for k in range(NZ):
            for j in range(NY):
                if j == 0 and k == 0: continue
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    var b = (e * nodes_per_elem + mid_node) * EULER_NC
                    var d = fabs(h[b + IDX_RHO] - rho_row[i])
                    if j != 0 and d > max_y_drift: max_y_drift = d
                    if k != 0 and d > max_z_drift: max_z_drift = d

    assert_true(not any_nan, "3D Sod NaN'd")
    print("max y-drift in ρ:", max_y_drift)
    print("max z-drift in ρ:", max_z_drift)
    assert_true(max_y_drift < Real(1.0e-10), "result not uniform in y")
    assert_true(max_z_drift < Real(1.0e-10), "result not uniform in z")

    var ix01 = Int(0.05 / mesh.dx)
    var ix06 = Int(0.6 / mesh.dx)
    var ix075 = Int(0.75 / mesh.dx)
    var ix095 = Int(0.95 / mesh.dx)
    print("ρ at x≈0.05 (left state ≈ 1.0):  ", rho_row[ix01])
    print("ρ at x≈0.6 (post-rarefaction ≈ 0.4263):", rho_row[ix06])
    print("ρ at x≈0.75 (post-contact ≈ 0.2655):  ", rho_row[ix075])
    print("ρ at x≈0.95 (right state ≈ 0.125):     ", rho_row[ix095])

    assert_true(fabs(rho_row[ix01] - Real(1.0)) < Real(1.0e-8), "left state off")
    assert_true(fabs(rho_row[ix06] - Real(0.42632)) < Real(4.0e-3), "post-rarefaction off")
    assert_true(fabs(rho_row[ix075] - Real(0.26557)) < Real(1.2e-3), "post-contact off")
    assert_true(fabs(rho_row[ix095] - Real(0.125)) < Real(1.0e-8), "right state off")

    # ---- Checks against the exact Riemann solution ---------------------
    var rho_L: Real = Real(1.0)
    var u_L: Real = Real(0.0)
    var p_L: Real = Real(1.0)
    var rho_R: Real = Real(0.125)
    var u_R: Real = Real(0.0)
    var p_R: Real = Real(0.1)
    var E_L = p_L / (GAMMA - Real(1.0)) + Real(0.5) * rho_L * u_L * u_L
    var E_R = p_R / (GAMMA - Real(1.0)) + Real(0.5) * rho_R * u_R * u_R

    # (1) Conservation with boundary-flux accounting. No wave has reached
    # x = 0 or x = 1, so the copy-out faces see the initial states and
    #   ∫U dV (t) = ∫U dV (0) + area · t · (F(U_L) − F(U_R))
    # holds exactly for U = (ρ, m_x, E).
    var cw = NodalBasis1D[P].weights()
    var c_area = Real(4 * NY * NZ) * mesh.jacobian_y * mesh.jacobian_z
    var c_jac = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var npe = (P + 1) * (P + 1) * (P + 1)
    var d_rho: Real = 0.0
    var d_mx: Real = 0.0
    var d_E: Real = 0.0
    var l1_rho: Real = 0.0
    var l1_u: Real = 0.0
    var l1_p: Real = 0.0
    var p_star = _star_pressure(rho_L, u_L, p_L, rho_R, u_R, p_R)
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            for n in range(npe):
                var w = c_jac
                var rem = n
                for _d in range(3):
                    w *= cw[rem % (P + 1)]
                    rem = rem // (P + 1)
                var b = (e * npe + n) * EULER_NC
                d_rho += w * (h[b + IDX_RHO] - host[b + IDX_RHO])
                d_mx += w * (h[b + IDX_MX] - host[b + IDX_MX])
                d_E += w * (h[b + IDX_E] - host[b + IDX_E])
        # (2) L1 error of the primitives against the exact solution, sampled
        # at the element centres of the first x-row (P = 2 mid-node).
        for i in range(NX):
            var b = (i * npe + npe // 2) * EULER_NC
            var x = (Real(i) + Real(0.5)) * mesh.dx
            var ex = _exact_riemann(
                rho_L, u_L, p_L, rho_R, u_R, p_R, p_star, (x - Real(0.5)) / t,
            )
            var r = h[b + IDX_RHO]
            var uu = h[b + IDX_MX] / r
            var pp = (GAMMA - Real(1.0)) * (h[b + IDX_E] - Real(0.5) * r * uu * uu)
            l1_rho += mesh.dx * fabs(r - ex[0])
            l1_u += mesh.dx * fabs(uu - ex[1])
            l1_p += mesh.dx * fabs(pp - ex[2])
    var err_mass = fabs(d_rho - c_area * t * (rho_L * u_L - rho_R * u_R)) / c_area
    var err_mom = fabs(
        d_mx - c_area * t * (
            rho_L * u_L * u_L + p_L - rho_R * u_R * u_R - p_R
        )
    ) / c_area
    var err_en = fabs(
        d_E - c_area * t * ((E_L + p_L) * u_L - (E_R + p_R) * u_R)
    ) / c_area
    print("  exact p* =", p_star)
    print("  conservation errors (mass, momentum, energy):",
          err_mass, err_mom, err_en)
    print("  L1 errors vs exact (ρ, u, p):", l1_rho, l1_u, l1_p)
    assert_true(err_mass < Real(1.0e-11), "mass balance violated")
    assert_true(err_mom < Real(1.0e-11), "momentum balance violated")
    assert_true(err_en < Real(1.0e-11), "energy balance violated")
    assert_true(l1_rho < Real(2.7e-2), "L1(ρ) vs exact Riemann solution too large")
    assert_true(l1_u < Real(5.0e-2), "L1(u) vs exact Riemann solution too large")
    assert_true(l1_p < Real(2.3e-2), "L1(p) vs exact Riemann solution too large")
    print("PASSED")
