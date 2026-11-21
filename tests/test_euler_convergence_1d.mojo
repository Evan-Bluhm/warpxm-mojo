# 1D 5-moment Euler DG convergence regression — confirm rate ≈ (P + 1)
# on the smooth entropy-wave problem. Extends the convergence-rate
# validation from linear scalar advection (`test_dg_convergence_1d`)
# and linear vector electromagnetic (`test_maxwell_convergence_1d`) to
# a **nonlinear** physics: Euler's momentum-flux is quadratic in `m_x`
# (via `m_x² / ρ`), so this test exercises a code path that the linear
# tests don't.
#
# IC + analytic solution:
#     ρ(x, 0) = 1 + ε sin(2π x)
#     u       = 1   (uniform)
#     P       = 1   (uniform)
# Pure entropy waves are linearly degenerate — `u` and `P` stay
# constant, only `ρ` advects at speed `u`. Exact solution at `t = T`:
#     ρ(x, T) = 1 + ε sin(2π (x − u T))
#
# Sweep: `P ∈ {1, 2, 3}` on `NX ∈ {32, 64}`. `T = 1` (one full period).
# `dt = 5·10⁻⁴` pinned to the finest-mesh CFL so SSPRK3's `O(dt³)`
# doesn't pollute.
#
# Build / run:
#     pixi run mojo run -I src tests/test_euler_convergence_1d.mojo

from std.math import log2, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime U: Real = 1.0
comptime DT: Real = 5.0e-4
comptime T_FINAL: Real = 1.0


def _one_run[P: Int](
    var ctx: DeviceContext, num_x: Int,
) raises -> Real:
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), num_x, periodic=True,
    )
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()

    var pp: Real = 1.0
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var rho = Real(1.0) + EPS * sin(TWO_PI * x)
            var ke = Real(0.5) * rho * U * U
            var b = (e * nodes_per_elem + i) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = rho * U
            host[b + IDX_MY] = Real(0.0)
            host[b + IDX_MZ] = Real(0.0)
            host[b + IDX_E] = pp / (GAMMA - Real(1.0)) + ke
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGEulerSolver1D[P](
        Euler(
            gas_gamma=GAMMA,
            min_density=Real(1e-12),
            min_pressure=Real(1e-12),
            cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState1D[P, EULER_NC]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    # After one period (u·T = 1), exact ρ returns to IC.
    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var exact = Real(1.0) + EPS * sin(TWO_PI * x)
                var b = (e * nodes_per_elem + i) * EULER_NC
                var d = h[b + IDX_RHO] - exact
                var jw = mesh.jacobian * weights[i]
                acc += jw * d * d
    return acc ** Real(0.5)


def _check_rate[P: Int](var ctx: DeviceContext, expected: Real) raises:
    var err_lo = _one_run[P](ctx, 32)
    var err_hi = _one_run[P](ctx, 64)
    var rate = log2(err_lo / err_hi)
    print(
        "P =", P,
        " err_NX32 =", err_lo,
        " err_NX64 =", err_hi,
        " rate =", rate,
    )
    assert_true(
        rate > expected - Real(0.25),
        "1D Euler entropy-wave convergence rate below tolerance",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0))
    _check_rate[2](ctx, Real(3.0))
    _check_rate[3](ctx, Real(4.0))
    print("PASSED")
