# 2D Euler hydrostatic-balance regression with uniform gravity.
#
# IC (a column under downward gravity):
#
#     ρ(x, y) = 1
#     u(x, y) = v(x, y) = 0
#     P(x, y) = P_0 - y                   (linear in y; P_0 chosen so P > 0)
#     E      = P / (γ - 1)
#     g      = (0, -1)
#
# Hydrostatic balance: ∂P/∂y = ρ · g_y, i.e. -1 = 1 · (-1). The continuous
# RHS for ρv is `-∂P/∂y + ρ·g_y = 1 + (-1) = 0`. With the strong-form DG
# discretization at P ≥ 1, the discrete ∂P/∂y is exact for linear P, so the
# gravity source kernel cancels the pressure-gradient flux divergence
# exactly. The state should sit still to machine precision.
#
# We integrate for ~50 timesteps and assert max|v|, max|u|, and the maximum
# deviation of ρ and E from their initial values all stay below 1e-12.

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 8
comptime NY: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime P0: Real = 2.0
comptime GX: Real = 0.0
comptime GY: Real = -1.0
comptime State = DGState[2, P, EULER_NC]


def _initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
    var p_local = P0 - y
    return [
        Real(1.0), Real(0.0), Real(0.0), Real(0.0),
        p_local / (GAMMA - Real(1.0)),
    ]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=False,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = State(ctx, mesh.num_elements)
    mesh.fill(q, _initial)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[2, P](app)
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_gravity_source(ctx, q_in, q_out, GX, GY)

    var dt: Real = 1.0e-3
    for _ in range(50):
        stepper.step(ctx, q, dt, compute_rhs, no_op_adjust[2, P, EULER_NC])
    ctx.synchronize()

    # Largest momenta, and largest departure of ρ and E from the IC.
    var max_v_abs: Real = 0.0
    var max_u_abs: Real = 0.0
    var max_rho_drift: Real = 0.0
    var max_e_drift: Real = 0.0
    def scan(
        x: Real, y: Real, w: Real, v: Array[Real, EULER_NC],
    ) {mut max_v_abs, mut max_u_abs, mut max_rho_drift, mut max_e_drift}:
        var v0 = _initial(x, y)
        var u_abs = fabs(v[IDX_MX])
        var v_abs = fabs(v[IDX_MY])
        var rho_d = fabs(v[IDX_RHO] - v0[IDX_RHO])
        var e_d = fabs(v[IDX_E] - v0[IDX_E])
        if u_abs > max_u_abs: max_u_abs = u_abs
        if v_abs > max_v_abs: max_v_abs = v_abs
        if rho_d > max_rho_drift: max_rho_drift = rho_d
        if e_d > max_e_drift: max_e_drift = e_d
    mesh.visit_nodes(q, scan)

    print("After 50 timesteps under hydrostatic balance:")
    print("  max |u|             =", max_u_abs)
    print("  max |v|             =", max_v_abs)
    print("  max |Δρ| vs IC      =", max_rho_drift)
    print("  max |ΔE| vs IC      =", max_e_drift)
    assert_true(max_u_abs < Real(1.0e-12), "horizontal velocity grew")
    assert_true(max_v_abs < Real(1.0e-12), "vertical velocity grew — hydrostatic balance broken")
    assert_true(max_rho_drift < Real(1.0e-12), "density drifted")
    assert_true(max_e_drift < Real(1.0e-12), "energy drifted")
    print("PASSED")
