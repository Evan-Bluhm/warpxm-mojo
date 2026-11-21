# Adaptive DP45 controller on a 2D Euler entropy wave.
#
# Smooth solution ρ(x, t) = 1 + 0.01·sin(2π(x − Ut)), uniform p and
# uniform u. Periodic mesh, t_final = 1 → solution returns to IC.
# Starts with a CFL-violating initial dt; controller must shrink it and
# then relax it. Validates the 2D port of `step_adaptive`.

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.dormand_prince_2d import DormandPrince45_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime U: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating; controller will shrink
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.125),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var rho = Real(1.0) + EPS * sin(TWO_PI * x)
                    var ke = Real(0.5) * rho * U * U
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * EULER_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = rho * U
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = Real(1.0) / (GAMMA - Real(1.0)) + ke
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var stepper = DormandPrince45_2D[P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = DT_INIT
    var num_steps: Int = 0
    var min_dt = dt
    var max_dt: Real = 0.0
    while t < T_FINAL - Real(1.0e-12):
        var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
        var ret = stepper.step_adaptive(
            ctx,
            q,
            dt_try,
            compute_rhs,
            atol=ATOL,
            rtol=RTOL,
        )
        var dt_used = ret[0]
        var dt_next = ret[1]
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break

    ctx.synchronize()

    # L2 error in ρ vs analytic IC (it returns to IC at t=1 since U=1, L=1).
    var rho_err: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * EULER_NC
                        var exact_rho = Real(1.0) + EPS * sin(TWO_PI * x)
                        var d = h[b + IDX_RHO] - exact_rho
                        var jw = mesh.jacobian_x * mesh.jacobian_y * weights[in_] * weights[jn]
                        rho_err += jw * d * d
    var l2 = rho_err ** Real(0.5)

    print("Adaptive DP45 2D entropy wave (P=2, NX=32, NY=4, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  ρ L2 error      :", l2)

    assert_true(num_steps < 500,
                "Adaptive DP45 took too many steps for smooth Euler entropy wave")
    assert_true(l2 < Real(1.0e-5),
                "Adaptive DP45 ρ error worse than expected")
    assert_true(max_dt > min_dt * Real(1.5),
                "Controller did not vary dt — likely a bug")
    print("PASSED")
