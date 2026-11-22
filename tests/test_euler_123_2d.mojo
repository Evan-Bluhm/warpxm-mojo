# 2D Einfeldt 1-2-3 problem — 1D 1-2-3 extruded along y.
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
# IC (Einfeldt et al. 1991):
#   Left  (x < 0.5): ρ = 1, u = -2, P = 0.4
#   Right (x ≥ 0.5): ρ = 1, u = +2, P = 0.4
#   γ = 1.4, t_final = 0.15

from std.collections import Array
from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 2
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.15
comptime DT: Real = 1.0e-4
comptime State = DGState[2, P, EULER_NC]


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
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=True,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var rho: Real = 1.0
        var u: Real = -2.0 if x < Real(0.5) else 2.0
        var pp: Real = 0.4
        var ke = Real(0.5) * rho * u * u
        return [rho, rho * u, Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0)) + ke]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[2, P](app)
    # The limiter tolerance is α = alpha_coeff · dx^1.5. The default
    # coefficient (1000) gives α ≈ 0.35, sixteen times the vacuum density,
    # so the limiter never acts there: density oscillations reach zero, the
    # positivity preserver pins a node at ρ_eps, and the sound speed at that
    # node (√(γP/ρ) ~ 10³) breaks the fixed time step. α ≈ 3.5e-3 keeps the
    # limiter active at the scale of the vacuum state.
    var limiter = MoeRossmanith[2, P, EULER_NC](
        ctx, mesh.num_elements, alpha_coeff=Real(10.0),
    )
    var positivity = PositivityEuler[2, P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    # --- diagnostics over every node of every element ---------------------
    var any_nan = False
    var min_rho: Real = 1.0e30
    var min_p: Real = 1.0e30
    var max_mt: Real = 0.0       # transverse momentum, exactly 0 in the IC
    var mass: Real = 0.0         # per unit transverse area
    var l1_rho: Real = 0.0       # per unit transverse area
    var area = (mesh.y_max - mesh.y_min)
    def scan(
        x: Real, y: Real, wq: Real, v: Array[Real, EULER_NC],
    ) {mut any_nan, mut min_rho, mut min_p, mut max_mt, mut mass, mut l1_rho, imm}:
        var rho = v[IDX_RHO]
        var mx = v[IDX_MX]
        var my = v[IDX_MY]
        var mz = v[IDX_MZ]
        var pp = (GAMMA - Real(1.0)) * (
            v[IDX_E] - Real(0.5) * (mx * mx + my * my + mz * mz) / rho
        )
        if isnan(rho) or isnan(pp): any_nan = True
        if rho < min_rho: min_rho = rho
        if pp < min_p: min_p = pp
        if fabs(my) > max_mt: max_mt = fabs(my)
        if fabs(mz) > max_mt: max_mt = fabs(mz)
        var jw = wq / area
        mass += jw * rho
        l1_rho += jw * fabs(rho - _exact_rho(x, t))
    mesh.visit_nodes(q, scan)

    comptime P1 = P + 1
    comptime nodes_per_elem = State.num_nodes_per_elem
    var w = NodalBasis1D[P].weights()
    var max_drift: Real = 0.0    # departure from the first transverse layer
    var rho_c: Real = 0.0
    var p_c: Real = 0.0
    var u_c: Real = 0.0
    with q.buf.map_to_host() as h:
        for et in range(NY):
            for i in range(NX):
                for n in range(nodes_per_elem):
                    for c in range(EULER_NC):
                        var d = fabs(
                            h[State.index(et * NX + i, n, c)]
                            - h[State.index(i, n, c)]
                        )
                        if d > max_drift: max_drift = d
        # Centre state: mean over the two elements adjacent to x = 1/2.
        for i in range(NX // 2 - 1, NX // 2 + 1):
            for n in range(nodes_per_elem):
                var rho = h[State.index(i, n, IDX_RHO)]
                var mx = h[State.index(i, n, IDX_MX)]
                var wgt = Real(0.5) * w[n % P1] * w[n // P1] / Real(4.0)
                rho_c += wgt * rho
                u_c += wgt * mx / rho
                p_c += wgt * (GAMMA - Real(1.0)) * (
                    h[State.index(i, n, IDX_E)] - Real(0.5) * mx * mx / rho
                )

    # Until the rarefaction heads reach the walls (t < 0.5 / (2 + c0) = 0.182)
    # the copy-out boundaries see the undisturbed state, so mass leaves at
    # exactly ρ0 u0 per unit area on each side.
    var mass_exact = Real(1.0) - Real(2.0) * Real(2.0) * t
    var rho_star: Real = 0.02185

    print("2D Einfeldt 1-2-3 problem — t =", t)
    print("  min ρ, min P (all nodes):", min_rho, min_p)
    print("  max transverse momentum: ", max_mt)
    print("  max transverse drift:    ", max_drift)
    print("  mass error:              ", fabs(mass - mass_exact))
    print("  L1 ρ error vs exact:     ", l1_rho)
    print("  centre ρ, u, P:          ", rho_c, u_c, p_c, " (exact 0.02185, 0, 0.001894)")

    assert_true(not any_nan, "2D 1-2-3 NaN'd")
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
