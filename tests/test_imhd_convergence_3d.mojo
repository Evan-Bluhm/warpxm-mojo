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
# Sweep: `P ∈ {2, 3}` × `NX ∈ {16, 32}`. `NY = NZ = 2` fixed slab.
# Then, at P = 2 and N = 16, the same wave along y and z with both the
# Rusanov and the HLLD flux (see `_check_axes`).
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


def _one_run[P: Int, HLLD: Bool](
    var ctx: DeviceContext, num: Int, axis: Int,
    drho_tol: Real, de_tol: Real,
) raises -> Real:
    """One Alfvén-wave run propagating along `axis` (0 = x, 1 = y, 2 = z)
    with `num` elements along it and a two-element slab across it.

    The setup is the x-propagating wave under a cyclic relabelling of the
    axes, `(x, y, z) -> (a0, a1, a2)`, which keeps it an exact solution and
    routes the non-trivial fluxes through the faces normal to `axis`.
    Returns the combined L² error of the four perturbation components.
    """
    var a0 = axis
    var a1 = (axis + 1) % 3
    var a2 = (axis + 2) % 3
    var nx = num if axis == 0 else NY
    var ny = num if axis == 1 else NY
    var nz = num if axis == 2 else NZ
    var mesh = BlockMesh3D[P](
        ctx,
        Real(0.0), Real(1.0) if axis == 0 else Real(1.0 / 4.0),
        Real(0.0), Real(1.0) if axis == 1 else Real(1.0 / 4.0),
        Real(0.0), Real(1.0) if axis == 2 else Real(1.0 / 4.0),
        nx=nx, ny=ny, nz=nz,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1
    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI
    # Element size along the propagation axis (the domain is [0, 1] there).
    var ds = Real(1.0) / Real(num)

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                var e = k * nx * ny + j * nx + i
                var cell = i if axis == 0 else (j if axis == 1 else k)
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var node = in_ if axis == 0 else (
                                jn if axis == 1 else kn
                            )
                            var s = (
                                Real(cell) * ds
                                + (Real(1.0) + ref_nodes[node]) * Real(0.5) * ds
                            )
                            var c_v = cos(k_wave * s)
                            var s_v = sin(k_wave * s)
                            var B1_ic = DELTA * c_v
                            var B2_ic = DELTA * s_v
                            var u1_ic = -DELTA * c_v * inv_sqrt_rho
                            var u2_ic = -DELTA * s_v * inv_sqrt_rho

                            var bsq = (
                                B0 * B0 + B1_ic * B1_ic + B2_ic * B2_ic
                            )
                            var msq = (
                                RHO0 * RHO0 * (u1_ic * u1_ic + u2_ic * u2_ic)
                            )
                            var E_total = (
                                P0 / (GAMMA - Real(1.0))
                                + Real(0.5) * msq / RHO0
                                + Real(0.5) * bsq
                            )

                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_NC
                            host[b + IDX_RHO] = RHO0
                            host[b + IDX_MX + a0] = Real(0.0)
                            host[b + IDX_MX + a1] = RHO0 * u1_ic
                            host[b + IDX_MX + a2] = RHO0 * u2_ic
                            host[b + IDX_E] = E_total
                            host[b + IDX_BX + a0] = B0
                            host[b + IDX_BX + a1] = B1_ic
                            host[b + IDX_BX + a2] = B2_ic
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGImhdSolver3D[P, HLLD](
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
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    var e = k * nx * ny + j * nx + i
                    var cell = i if axis == 0 else (j if axis == 1 else k)
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var node = in_ if axis == 0 else (
                                    jn if axis == 1 else kn
                                )
                                var s = (
                                    Real(cell) * ds
                                    + (Real(1.0) + ref_nodes[node])
                                      * Real(0.5) * ds
                                )
                                var c_v = cos(k_wave * s)
                                var s_v = sin(k_wave * s)
                                var B1_ex = -DELTA * c_v
                                var B2_ex = -DELTA * s_v
                                var u1_ex = DELTA * c_v * inv_sqrt_rho
                                var u2_ex = DELTA * s_v * inv_sqrt_rho
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MHD_NC
                                var rho = h[b + IDX_RHO]
                                var u1 = h[b + IDX_MX + a1] / rho
                                var u2 = h[b + IDX_MX + a2] / rho
                                var d_B1 = h[b + IDX_BX + a1] - B1_ex
                                var d_B2 = h[b + IDX_BX + a2] - B2_ex
                                var d_u1 = u1 - u1_ex
                                var d_u2 = u2 - u2_ex
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
                                    d_B1 * d_B1 + d_B2 * d_B2
                                    + d_u1 * d_u1 + d_u2 * d_u2
                                )
    print(
        "  P =", P, " HLLD =", HLLD, " axis =", axis, " N =", num,
        " max |Δρ| =", max_drho, " max |ΔE| =", max_dE,
    )
    assert_true(max_drho < drho_tol, "density drifted (Alfvén wave is incompressible)")
    assert_true(max_dE < de_tol, "total energy drifted")
    return acc ** Real(0.5)


def _check_rate[P: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
    drho_tol: Real, de_tol: Real,
) raises -> Real:
    """Check the observed rate and bound both errors absolutely. Returns
    the coarse-mesh error.

    `tol_lo` / `tol_hi` are about 3x the measured coarse / fine errors, so
    a defect that scales the error uniformly (and leaves the rate intact)
    still fails.
    """
    var err_lo = _one_run[P, False](ctx, 16, 0, drho_tol, de_tol)
    var err_hi = _one_run[P, False](ctx, 32, 0, drho_tol, de_tol)
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
    return err_lo


def _rel_diff(a: Real, b: Real) -> Real:
    var d = a - b
    if d < Real(0.0): d = -d
    return d / b


def _check_axes(var ctx: DeviceContext, rus_x: Real) raises:
    """HLLD accuracy, and every face-normal direction of both fluxes.

    The x-propagating wave only sends non-trivial states through x-normal
    faces. Re-running it along y and z (a cyclic relabelling of the axes)
    exercises the y- and z-normal paths of the flux and of the DG core;
    the error must match the x run, because the discretisation is
    symmetric under that relabelling. `rus_x` is the Rusanov P=2, N=16
    error of the x-propagating run from the rate check.
    """
    var rus_y = _one_run[2, False](ctx, 16, 1, Real(3.3e-4), Real(2.0e-5))
    var rus_z = _one_run[2, False](ctx, 16, 2, Real(3.3e-4), Real(2.0e-5))
    var hlld_x = _one_run[2, True](ctx, 16, 0, Real(4.0e-4), Real(2.0e-5))
    var hlld_y = _one_run[2, True](ctx, 16, 1, Real(4.0e-4), Real(2.0e-5))
    var hlld_z = _one_run[2, True](ctx, 16, 2, Real(4.0e-4), Real(2.0e-5))
    print("Rusanov P=2 N=16  x / y / z:", rus_x, rus_y, rus_z)
    print("HLLD    P=2 N=16  x / y / z:", hlld_x, hlld_y, hlld_z)
    # Measured 4.07e-5 (Rusanov: 4.67e-5).
    assert_true(hlld_x < Real(1.2e-4), "HLLD L2 error above tolerance")
    assert_true(
        hlld_x < rus_x,
        "HLLD should be less dissipative than Rusanov on a smooth wave",
    )
    # Axis symmetry holds to round-off (measured relative spread ~3e-14).
    var sym_tol = Real(1.0e-9)
    assert_true(
        _rel_diff(rus_y, rus_x) < sym_tol and _rel_diff(rus_z, rus_x) < sym_tol,
        "Rusanov error differs between propagation axes",
    )
    assert_true(
        _rel_diff(hlld_y, hlld_x) < sym_tol
        and _rel_diff(hlld_z, hlld_x) < sym_tol,
        "HLLD error differs between propagation axes",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    # P=1 omitted: at NX=16, P=1 is borderline for the asymptotic
    # h^(P+1) regime on the smooth Alfvén-wave problem.
    var rus_x = _check_rate[2](ctx, Real(3.0), Real(1.4e-4), Real(1.8e-5),
        Real(3.3e-4), Real(2.0e-5))
    _ = _check_rate[3](ctx, Real(4.0), Real(2.3e-6), Real(1.4e-7),
        Real(1.7e-5), Real(5.0e-7))
    _check_axes(ctx, rus_x)
    print("PASSED")
