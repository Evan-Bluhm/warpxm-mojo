# 3D ideal-MHD DG convergence regression — confirm rate ≈ (P + 1) on
# the smooth circularly-polarised Alfvén wave, run on a 3D mesh.
# Mirrors `test_imhd_convergence_2d` extended to the 3D core.
#
# This is the final entry in the convergence-test family (1D / 2D / 3D
# all covered for ≥ NC = 1; 1D additionally validates NC = 5 (Euler)
# and NC = 6 (Maxwell); 2D validates NC = 8 (MHD); this test validates
# the 3D core (`spatial/dg_core_3d.mojo`) preserves theoretical order
# at NC = 8 — important because the 3D core uses the single-
# accumulator pattern from DG-perf Task 2 (different code shape from
# the 2D core's surface-array layout).
#
# Setup (uniform in y, z; ky = kz = 0, so 1D-in-x propagation on a
# thin 3D slab):
#     ρ = 1, P = 1, B = (1, δ cos(2π x), δ sin(2π x))
#     u = (0, -δ cos(2π x) / √ρ, -δ sin(2π x) / √ρ)   [Walen]
#
# After T = λ/c_A = 1 the solution returns to the IC; the test stops at
# T/2, where it is the IC with the transverse perturbation negated.
#
# Sweep: `P ∈ {1, 2, 3}` × `NX ∈ {16, 32}`. `NY = NZ = 2` fixed slab.
# CFL pinned to `dt = 5·10⁻⁴`.
#
# Note: smaller NX values than the 2D test (16 / 32 vs 32 / 64). The
# 3D state buffer scales as `NX · NY · NZ · (P+1)³`, so 3D runs cost
# `(NY · NZ · (P+1))` ≈ 50 × more per cell than 2D. The 16 / 32 sweep
# is the cheapest pair that gets us into
# the asymptotic regime for P = 2, 3 (P = 1 is marginal — accepted).
#
# Build / run:
#     pixi run mojo run -I src tests/test_imhd_convergence_3d.mojo

from std.math import cos, log2, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime NY: Int = 2
comptime NZ: Int = 2
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO0: Real = 1.0
comptime P0: Real = 1.0
comptime B0: Real = 1.0
comptime DELTA: Real = 0.1
comptime DT: Real = 5.0e-4
# Half a period: the wave has moved by λ/2, so the exact solution is the
# IC with the transverse perturbation negated. Half the steps of a full
# period, same convergence rates.
comptime T_FINAL: Real = 0.5


def _one_run[P: Int](
    var ctx: DeviceContext, num_x: Int,
    drho_tol: Real, de_tol: Real,
) raises -> Real:
    var mesh = BlockMesh3D[P](
        ctx,
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0 / 4.0),
        Real(0.0), Real(1.0 / 4.0),
        nx=num_x, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1
    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            for i in range(num_x):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * num_x * NY + j * num_x + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var x = (
                                xl
                                + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            )
                            var c_v = cos(k_wave * x)
                            var s_v = sin(k_wave * x)
                            var By_ic = DELTA * c_v
                            var Bz_ic = DELTA * s_v
                            var uy_ic = -DELTA * c_v * inv_sqrt_rho
                            var uz_ic = -DELTA * s_v * inv_sqrt_rho

                            var bsq = (
                                B0 * B0 + By_ic * By_ic + Bz_ic * Bz_ic
                            )
                            var msq = (
                                RHO0 * RHO0 * (uy_ic * uy_ic + uz_ic * uz_ic)
                            )
                            var E_total = (
                                P0 / (GAMMA - Real(1.0))
                                + Real(0.5) * msq / RHO0
                                + Real(0.5) * bsq
                            )

                            var n = kn * P1 * P1 + jn * P1 + in_
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

    var spatial = DGImhdSolver3D[P](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3_3D[P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState3D[P, MHD_NC],
        mut q_out: DGState3D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState3D[P, MHD_NC]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    # Combined L² over the four perturbation components.
    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var acc: Real = 0.0
    var max_drho: Real = 0.0
    var max_dE: Real = 0.0
    # ρ and E are uniform in the exact solution (the wave is incompressible).
    var E_exact = (
        P0 / (GAMMA - Real(1.0))
        + Real(0.5) * (B0 * B0 + DELTA * DELTA)
        + Real(0.5) * RHO0 * DELTA * DELTA * inv_sqrt_rho * inv_sqrt_rho
    )
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(num_x):
                    var xl = mesh.x_min + Real(i) * mesh.dx
                    var e = k * num_x * NY + j * num_x + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var x = (
                                    xl
                                    + (Real(1.0) + ref_nodes[in_])
                                      * mesh.jacobian_x
                                )
                                var c_v = cos(k_wave * x)
                                var s_v = sin(k_wave * x)
                                var By_ex = -DELTA * c_v
                                var Bz_ex = -DELTA * s_v
                                var uy_ex = DELTA * c_v * inv_sqrt_rho
                                var uz_ex = DELTA * s_v * inv_sqrt_rho
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MHD_NC
                                var rho = h[b + IDX_RHO]
                                var uy = h[b + IDX_MY] / rho
                                var uz = h[b + IDX_MZ] / rho
                                var d_By = h[b + IDX_BY] - By_ex
                                var d_Bz = h[b + IDX_BZ] - Bz_ex
                                var d_uy = uy - uy_ex
                                var d_uz = uz - uz_ex
                                var d_rho = rho - RHO0
                                var d_E = h[b + IDX_E] - E_exact
                                if d_rho > max_drho: max_drho = d_rho
                                if -d_rho > max_drho: max_drho = -d_rho
                                if d_E > max_dE: max_dE = d_E
                                if -d_E > max_dE: max_dE = -d_E
                                var jw = (
                                    jv * weights[in_] * weights[jn] * weights[kn]
                                )
                                acc += jw * (
                                    d_By * d_By + d_Bz * d_Bz
                                    + d_uy * d_uy + d_uz * d_uz
                                )
    print("  P =", P, " NX =", num_x, " max |Δρ| =", max_drho, " max |ΔE| =", max_dE)
    assert_true(max_drho < drho_tol, "density drifted (Alfvén wave is incompressible)")
    assert_true(max_dE < de_tol, "total energy drifted")
    return acc ** Real(0.5)


def _check_rate[P: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
    drho_tol: Real, de_tol: Real,
) raises:
    """Check the observed rate and bound both errors absolutely.

    `tol_lo` / `tol_hi` are about 3x the measured coarse / fine errors, so
    a defect that scales the error uniformly (and leaves the rate intact)
    still fails.
    """
    var err_lo = _one_run[P](ctx, 16, drho_tol, de_tol)
    var err_hi = _one_run[P](ctx, 32, drho_tol, de_tol)
    var rate = log2(err_lo / err_hi)
    assert_true(err_lo < tol_lo, "coarse-mesh L2 error above tolerance")
    assert_true(err_hi < tol_hi, "fine-mesh L2 error above tolerance")
    print(
        "P =", P,
        " err_NX16 =", err_lo,
        " err_NX32 =", err_hi,
        " rate =", rate,
    )
    assert_true(
        rate > expected - Real(0.25),
        "3D MHD Alfvén-wave convergence rate below tolerance",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    # P=1 omitted: at NX=16, P=1 is borderline for the asymptotic
    # h^(P+1) regime on the smooth Alfvén-wave problem.
    _check_rate[2](ctx, Real(3.0), Real(1.4e-4), Real(1.8e-5),
        Real(3.3e-4), Real(2.0e-5))
    _check_rate[3](ctx, Real(4.0), Real(2.3e-6), Real(1.4e-7),
        Real(1.7e-5), Real(5.0e-7))
    print("PASSED")
