# 2D ideal-MHD DG convergence regression — confirm rate ≈ (P + 1) on
# the smooth circularly-polarised Alfvén wave (the canonical smooth
# MHD test problem). Mirrors the layout of `test_imhd_alfven_2d`
# but sweeps over `P` and `NX`.
#
# This is the MHD entry in the convergence-test family alongside
# `test_dg_convergence_{1,2,3}d` (scalar advection),
# `test_maxwell_convergence_1d` (linear vector EM), and
# `test_euler_convergence_1d` (nonlinear hydrodynamics).
#
# Setup (uniform in y, ky=0, so really 1D-in-x propagation on a thin
# 2D slab):
#     ρ = 1, P = 1, B = (1, δ cos(2π x), δ sin(2π x))
#     u = (0, -δ cos(2π x) / √ρ, -δ sin(2π x) / √ρ)   [Walen relation]
#
# Alfvén speed c_A = B_x / √ρ = 1, so after T = λ/c_A = 1 the solution
# returns to the IC.
#
# Sweep: `P ∈ {1, 2, 3}` × `NX ∈ {32, 64}`, `NY = 4` fixed (thin slab
# just for the 2D solver). CFL pinned to `dt = 5·10⁻⁴` across the
# whole sweep.
#
# Build / run:
#     pixi run mojo run -I src tests/test_imhd_convergence_2d.mojo

from std.math import cos, log2, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO0: Real = 1.0
comptime P0: Real = 1.0
comptime B0: Real = 1.0
comptime DELTA: Real = 0.1
comptime DT: Real = 5.0e-4
comptime T_FINAL: Real = 1.0


def _one_run[P: Int](
    var ctx: DeviceContext, num_x: Int,
) raises -> Real:
    var mesh = BlockMesh2D[P](
        ctx,
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0 / 8.0),
        nx=num_x, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(num_x):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * num_x + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = (
                        xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    )
                    var c_v = cos(k_wave * x)
                    var s_v = sin(k_wave * x)
                    var By_ic = DELTA * c_v
                    var Bz_ic = DELTA * s_v
                    var uy_ic = -DELTA * c_v * inv_sqrt_rho
                    var uz_ic = -DELTA * s_v * inv_sqrt_rho

                    var bsq = B0 * B0 + By_ic * By_ic + Bz_ic * Bz_ic
                    var msq = (
                        RHO0 * RHO0 * (uy_ic * uy_ic + uz_ic * uz_ic)
                    )
                    var E_total = (
                        P0 / (GAMMA - Real(1.0))
                        + Real(0.5) * msq / RHO0
                        + Real(0.5) * bsq
                    )

                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_NC
                    host[b + IDX_RHO] = RHO0
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = RHO0 * uy_ic
                    host[b + IDX_MZ] = RHO0 * uz_ic
                    host[b + IDX_E] = E_total
                    host[b + IDX_BX] = B0
                    host[b + IDX_BY] = By_ic
                    host[b + IDX_BZ] = Bz_ic
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGImhdSolver2D[P](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState2D[P, MHD_NC]) raises:
        pass

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    # Combined L² over the four perturbation components (B_y, B_z, u_y, u_z).
    # After exactly one period, the exact solution equals the IC.
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(num_x):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * num_x + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var x = (
                            xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        )
                        var c_v = cos(k_wave * x)
                        var s_v = sin(k_wave * x)
                        var By_ex = DELTA * c_v
                        var Bz_ex = DELTA * s_v
                        var uy_ex = -DELTA * c_v * inv_sqrt_rho
                        var uz_ex = -DELTA * s_v * inv_sqrt_rho
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_NC
                        var rho = h[b + IDX_RHO]
                        var uy = h[b + IDX_MY] / rho
                        var uz = h[b + IDX_MZ] / rho
                        var d_By = h[b + IDX_BY] - By_ex
                        var d_Bz = h[b + IDX_BZ] - Bz_ex
                        var d_uy = uy - uy_ex
                        var d_uz = uz - uz_ex
                        var jw = ja * weights[in_] * weights[jn]
                        acc += jw * (
                            d_By * d_By + d_Bz * d_Bz
                            + d_uy * d_uy + d_uz * d_uz
                        )
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
        "2D MHD Alfvén-wave convergence rate below tolerance",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0))
    _check_rate[2](ctx, Real(3.0))
    _check_rate[3](ctx, Real(4.0))
    print("PASSED")
