# 1D3V Vlasov between two specular walls (`BC_VLASOV_REFLECT`).
#
# With no fields a particle streams at constant velocity and reverses
# `v_1` each time it hits `x = 0` or `x = 1`. By the method of images the
# box is the periodic line of period 2 on which the data at `2 − x` are
# the data at `x` with `v_1 → −v_1`. The initial condition
#
#     f₀(x, v) = (1 + ε cos πx) · exp(−[(v_1 − U sin πx)² + v_2² + v_3²] / 2σ²)
#
# is already that extension — replacing `x` by `2 − x` and `v_1` by `−v_1`
# leaves the formula unchanged — so it is consistent with the walls (it
# is even in `v_1` at both of them) and the exact solution is simply
#
#     f(x, v, t) = f₀(x − v_1 t, v)
#
# with the formula evaluated outside `[0, 1]`. The drift `U sin πx` makes
# `f` strongly asymmetric in `v_1` in the interior, so a wall that copies
# the interior state (or lets it leave) instead of mirroring it is an
# O(1) error. By `t = 0.6` the fastest particles (`|v_1| = 4`) have
# crossed the box more than twice.
#
# Checks:
#   (a) L² error against the exact solution, order `P_x + 1` in the x
#       mesh. Velocity nodes are only coupled through the walls, where
#       the mirror of a node is a node, so the velocity mesh adds no
#       error;
#   (b) total mass conserved to round-off: the upwind flux at `v_1` and
#       at `−v_1` cancel exactly at each wall node;
#   (c) the mesh refuses a specular wall when the velocity mesh is not
#       symmetric about zero.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_reflecting_wall_1d3v.mojo

from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime P_V: Int = 2
comptime NV1: Int = 8
comptime NV_PERP: Int = 3
comptime V_MAX: Real = 4.0
comptime EPS: Real = 0.3
comptime DRIFT: Real = 0.8
comptime INV_2SIGMA_SQ: Real = 1.0 / (2.0 * 0.7 * 0.7)
comptime T_FINAL: Real = 0.6


def f_image(y: Real, v1: Real, v2: Real, v3: Real) -> Real:
    """`f₀`, valid for any real `y` (the image extension)."""
    var w1 = v1 - DRIFT * sin(PI * y)
    return (1.0 + EPS * cos(PI * y)) * exp(
        -INV_2SIGMA_SQ * (w1 * w1 + v2 * v2 + v3 * v3)
    )


def wall_run[P_X: Int](var ctx: DeviceContext, nx: Int) raises -> Real:
    """Relative L² error at `T_FINAL`; asserts mass conservation."""
    comptime State = PhaseState[1, 3, P_X, P_V, VLASOV_NC]
    var mesh = PhaseMesh[1, 3, P_X, P_V](
        ctx,
        lo=[0.0, -V_MAX, -V_MAX, -V_MAX],
        hi=[1.0, V_MAX, V_MAX, V_MAX],
        n=[nx, NV1, NV_PERP, NV_PERP],
        periodic=[False],
        bc=[BC_VLASOV_REFLECT, BC_VLASOV_REFLECT],
    )
    var q = State(ctx, mesh.num_phase_elements)
    mesh.fill(q, f_image)
    var mass0 = mesh.integrate(q)

    var spatial = DGVlasovSolver[1, 3, P_X, P_V](VlasovMaxwell(cfl_max=Real(0.5)))
    var stepper = SSPRK3Phase[1, 3, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    var em = DGState[1, P_X, MAXWELL_NC](ctx, nx)   # zero fields

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, em, q_out)
    def no_adjust(mut q_: State) raises:
        pass
    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, em)
    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), T_FINAL, compute_rhs, no_adjust, suggest_dt,
        dt_interval=1000000,
    )
    ctx.synchronize()

    var mass1 = mesh.integrate(q)
    var dm = (mass1 - mass0) / mass0
    print("  NX =", nx, " relative mass change =", dm)
    assert_true(abs(dm) < Real(1.0e-12), "specular walls did not conserve mass")

    var err_sq: Real = 0.0
    var norm_sq: Real = 0.0
    def add(
        x: Real, v1: Real, v2: Real, v3: Real, w: Real, f: Real,
    ) {mut err_sq, mut norm_sq}:
        var exact = f_image(x - v1 * T_FINAL, v1, v2, v3)
        err_sq += w * (f - exact) * (f - exact)
        norm_sq += w * exact * exact
    mesh.visit_nodes(q, add)
    return sqrt(err_sq / norm_sq)


def check_rejects_asymmetric_velocity_mesh(var ctx: DeviceContext) raises:
    var raised = False
    try:
        var mesh = PhaseMesh[1, 3, 1, 1](
            ctx,
            lo=[0.0, -3.0, -V_MAX, -V_MAX],
            hi=[1.0, V_MAX, V_MAX, V_MAX],
            n=[2, 2, 2, 2],
            periodic=[False],
            bc=[BC_VLASOV_REFLECT, BC_COPYOUT],
        )
        _ = mesh.num_phase_elements
    except e:
        raised = True
    assert_true(raised, "asymmetric velocity mesh accepted a specular wall")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    check_rejects_asymmetric_velocity_mesh(ctx)

    # Measured: 5.73e-3 → 1.48e-3 (rate 1.95) and 3.04e-4 → 3.69e-5 (rate
    # 3.04); tolerances are about 1.5x that. With copy-out walls in place
    # of the specular ones the error is 0.4 at every resolution and a
    # quarter of the mass leaves. With the specular walls the mass
    # changes by less than 1e-13.
    var lo1 = wall_run[1](ctx, 32)
    var hi1 = wall_run[1](ctx, 64)
    _ = check_convergence(
        "1D3V specular walls P_x = 1 (NX 32 → 64)",
        lo1, hi1, Real(2.0), Real(9.0e-3), Real(2.3e-3),
    )
    var lo2 = wall_run[2](ctx, 16)
    var hi2 = wall_run[2](ctx, 32)
    _ = check_convergence(
        "1D3V specular walls P_x = 2 (NX 16 → 32)",
        lo2, hi2, Real(3.0), Real(5.0e-4), Real(6.0e-5),
    )
    print("PASSED")
