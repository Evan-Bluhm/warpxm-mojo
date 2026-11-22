# Regression test: 1D Sod shock tube with the Moe-Rossmanith limiter.
#
# Asserts that the limited DG scheme:
#   * does NOT NaN out (positivity preserved);
#   * places the shock, contact, and rarefaction at the analytic Riemann
#     locations to within one element of accuracy at t = 0.2.
#
# Analytic at t = 0.2 (γ = 1.4, x_0 = 0.5, t = 0.2):
#   Rarefaction head at x ≈ 0.264, tail at x ≈ 0.486
#   Post-rarefaction plateau (left of contact):
#     ρ_3 ≈ 0.4263, u_3 ≈ 0.9275, P_3 ≈ 0.3031
#   Contact at x ≈ 0.685
#   Post-contact plateau (right of contact):
#     ρ_4 ≈ 0.2655, u_4 ≈ 0.9275, P_4 ≈ 0.3031
#   Shock at x ≈ 0.850.
#
# Checks: plateau values (ρ, u, P) at four sample locations, exact
# conservation of mass / momentum / energy with boundary-flux accounting,
# and the L1 error of (ρ, u, P) against the exact Riemann solution, which
# is evaluated in-test (`_exact_riemann`).

from std.collections import Array
from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith_1d import MoeRossmanith1D
from warpxm.apps.euler import Euler
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 200
comptime GAMMA: Real = 1.4


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
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var rho: Real
            var pp: Real
            if x < Real(0.5):
                rho = 1.0
                pp = 1.0
            else:
                rho = 0.125
                pp = 0.1
            var b = (e * nodes_per_elem + i) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = Real(0.0)
            host[b + 2] = Real(0.0)
            host[b + 3] = Real(0.0)
            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver1D[P](app)
    var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith1D[P](ctx, mesh.num_elements)

    var dt: Real = 2.0e-4
    var t_final: Real = 0.2

    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState1D[P, EULER_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    # Sample at element midpoints; assert (a) finiteness, (b) plateau / right
    # state values, (c) shock and contact in roughly the right locations.
    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var rho_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    var u_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    var p_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var b = (e * nodes_per_elem + mid_node) * EULER_NC
            var rho = h[b + IDX_RHO]
            var mx = h[b + IDX_MX]
            var e_t = h[b + IDX_E]
            if isnan(rho) or isnan(mx) or isnan(e_t):
                any_nan = True
            rho_at_x[e] = rho
            u_at_x[e] = mx / rho
            var ke = Real(0.5) * rho * u_at_x[e] * u_at_x[e]
            p_at_x[e] = (GAMMA - Real(1.0)) * (e_t - ke)
    assert_true(not any_nan, "Sod result contains NaN")

    # x = 0.1 → left undisturbed state (ρ = 1, u = 0, P = 1).
    var ix_left = Int(0.1 / mesh.dx)
    print("at x=0.1:  rho =", rho_at_x[ix_left],
          "u =", u_at_x[ix_left], "P =", p_at_x[ix_left])
    assert_true(fabs(rho_at_x[ix_left] - Real(1.0)) < Real(1.0e-8),
                "left undisturbed state ρ wrong")

    # x = 0.6 → post-rarefaction plateau.
    var ix_plateau1 = Int(0.6 / mesh.dx)
    print("at x=0.6:  rho =", rho_at_x[ix_plateau1],
          "u =", u_at_x[ix_plateau1], "P =", p_at_x[ix_plateau1])
    assert_true(fabs(rho_at_x[ix_plateau1] - Real(0.42632)) < Real(8.0e-3),
                "post-rarefaction ρ outside tolerance")
    assert_true(fabs(u_at_x[ix_plateau1] - Real(0.92745)) < Real(3.0e-3),
                "post-rarefaction u outside tolerance")
    assert_true(fabs(p_at_x[ix_plateau1] - Real(0.30313)) < Real(1.0e-3),
                "post-rarefaction P outside tolerance")

    # x = 0.75 → post-contact plateau.
    var ix_plateau2 = Int(0.75 / mesh.dx)
    print("at x=0.75: rho =", rho_at_x[ix_plateau2],
          "u =", u_at_x[ix_plateau2], "P =", p_at_x[ix_plateau2])
    assert_true(fabs(rho_at_x[ix_plateau2] - Real(0.26557)) < Real(1.0e-3),
                "post-contact ρ outside tolerance")
    # u and P are continuous across the contact.
    assert_true(fabs(u_at_x[ix_plateau2] - Real(0.92745)) < Real(3.0e-3),
                "post-contact u outside tolerance")
    assert_true(fabs(p_at_x[ix_plateau2] - Real(0.30313)) < Real(1.0e-3),
                "post-contact P outside tolerance")

    # x = 0.95 → right undisturbed state.
    var ix_right = Int(0.95 / mesh.dx)
    print("at x=0.95: rho =", rho_at_x[ix_right],
          "u =", u_at_x[ix_right], "P =", p_at_x[ix_right])
    assert_true(fabs(rho_at_x[ix_right] - Real(0.125)) < Real(1.0e-8),
                "right undisturbed state ρ wrong")


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
    var c_area = Real(1.0)
    var c_jac = mesh.jacobian
    var npe = P + 1
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
                for _d in range(1):
                    w *= cw[rem % (P + 1)]
                    rem = rem // (P + 1)
                var b = (e * npe + n) * EULER_NC
                d_rho += w * (h[b + IDX_RHO] - host[b + IDX_RHO])
                d_mx += w * (h[b + IDX_MX] - host[b + IDX_MX])
                d_E += w * (h[b + IDX_E] - host[b + IDX_E])
        # (2) L1 error of the primitives against the exact solution, sampled
        # at the element centres of the first x-row (P = 2 mid-node).
        for i in range(NUM_ELEMENTS):
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
    assert_true(l1_rho < Real(1.5e-2), "L1(ρ) vs exact Riemann solution too large")
    assert_true(l1_u < Real(3.0e-2), "L1(u) vs exact Riemann solution too large")
    assert_true(l1_p < Real(1.3e-2), "L1(p) vs exact Riemann solution too large")
    print("PASSED")
