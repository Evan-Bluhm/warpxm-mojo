# 2D ideal-MHD DG convergence regression — confirm rate ≈ (P + 1) on
# the smooth circularly-polarised Alfvén wave (the canonical smooth
# MHD test problem), sweeping over `P` and `NX`. Every run also asserts
# an absolute L2 bound and that ρ and E stay uniform.
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
# returns to the IC; the test stops at T/2, where it is the IC with the
# transverse perturbation negated.
#
# After the Rusanov sweep, P = 2 is repeated with the HLLD flux, and the
# P = 2, N = 32 case is re-run propagating along y with both fluxes
# (see `_check_y_axis`).
#
# Sweep: `P ∈ {1, 2, 3}` × `NX ∈ {32, 64}`, `NY = 2` fixed (thin slab
# just for the 2D solver). CFL pinned to `dt = 5·10⁻⁴` across the
# whole sweep.
#
# Build / run:
#     pixi run mojo run -I src tests/test_imhd_convergence_2d.mojo

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd import *
from warpxm.testing.convergence import check_convergence


comptime NY: Int = 2
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
    """One Alfvén-wave run propagating along `axis` (0 = x, 1 = y) with
    `num` elements along it and a two-element slab across it.

    The setup is the x-propagating wave under a cyclic relabelling of the
    vector components, `(x, y, z) -> (a0, a1, a2)`, which keeps it an exact
    solution and routes the non-trivial fluxes through the faces normal to
    `axis`. Returns the combined L² error of the four perturbation
    components.
    """
    var a0 = axis
    var a1 = (axis + 1) % 3
    var a2 = (axis + 2) % 3
    var nx = num if axis == 0 else NY
    var ny = num if axis == 1 else NY
    var mesh = BlockMesh[2, P](
        ctx,
        Real(0.0), Real(1.0) if axis == 0 else Real(1.0 / 8.0),
        Real(0.0), Real(1.0) if axis == 1 else Real(1.0 / 8.0),
        nx=nx, ny=ny, periodic_x=True, periodic_y=True,
    )
    comptime State = DGState[2, P, MHD_NC]
    var q = State(ctx, mesh.num_elements)

    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    def initial(x: Real, y: Real) {imm} -> Array[Real, MHD_NC]:
        var s = x if axis == 0 else y
        var c_v = cos(k_wave * s)
        var s_v = sin(k_wave * s)
        var B1_ic = DELTA * c_v
        var B2_ic = DELTA * s_v
        var u1_ic = -DELTA * c_v * inv_sqrt_rho
        var u2_ic = -DELTA * s_v * inv_sqrt_rho
        var bsq = B0 * B0 + B1_ic * B1_ic + B2_ic * B2_ic
        var msq = RHO0 * RHO0 * (u1_ic * u1_ic + u2_ic * u2_ic)
        var v = Array[Real, MHD_NC](fill=Real(0.0))
        v[IDX_RHO] = RHO0
        v[IDX_MX + a1] = RHO0 * u1_ic
        v[IDX_MX + a2] = RHO0 * u2_ic
        v[IDX_E] = (
            P0 / (GAMMA - Real(1.0))
            + Real(0.5) * msq / RHO0
            + Real(0.5) * bsq
        )
        v[IDX_BX + a0] = B0
        v[IDX_BX + a1] = B1_ic
        v[IDX_BX + a2] = B2_ic
        return v^
    mesh.fill(q, initial)

    var spatial = DGImhdSolver[2, P, HLLD](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3[2, P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT,
        compute_rhs, no_op_adjust[2, P, MHD_NC],
    )
    ctx.synchronize()

    # Combined L² over the four perturbation components. At T/2 the exact
    # solution is the IC with the transverse perturbation negated.
    var acc: Real = 0.0
    var max_drho: Real = 0.0
    var max_dE: Real = 0.0
    # ρ and E are uniform in the exact solution (the wave is incompressible).
    var E_exact = (
        P0 / (GAMMA - Real(1.0))
        + Real(0.5) * (B0 * B0 + DELTA * DELTA)
        + Real(0.5) * RHO0 * DELTA * DELTA * inv_sqrt_rho * inv_sqrt_rho
    )
    def error(
        x: Real, y: Real, w: Real, v: Array[Real, MHD_NC],
    ) {mut acc, mut max_drho, mut max_dE, imm}:
        var s = x if axis == 0 else y
        var c_v = cos(k_wave * s)
        var s_v = sin(k_wave * s)
        var rho = v[IDX_RHO]
        var d_B1 = v[IDX_BX + a1] + DELTA * c_v
        var d_B2 = v[IDX_BX + a2] + DELTA * s_v
        var d_u1 = v[IDX_MX + a1] / rho - DELTA * c_v * inv_sqrt_rho
        var d_u2 = v[IDX_MX + a2] / rho - DELTA * s_v * inv_sqrt_rho
        max_drho = fmax(max_drho, fabs(rho - RHO0))
        max_dE = fmax(max_dE, fabs(v[IDX_E] - E_exact))
        acc += w * (d_B1 * d_B1 + d_B2 * d_B2 + d_u1 * d_u1 + d_u2 * d_u2)
    mesh.visit_nodes(q, error)
    print(
        "  P =", P, " HLLD =", HLLD, " axis =", axis, " N =", num,
        " max |Δρ| =", max_drho, " max |ΔE| =", max_dE,
    )
    assert_true(max_drho < drho_tol, "density drifted (Alfvén wave is incompressible)")
    assert_true(max_dE < de_tol, "total energy drifted")
    return sqrt(acc)


def _check_rate[P: Int, HLLD: Bool](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
    drho_tol: Real, de_tol: Real,
) raises -> Real:
    """Check the observed rate between N = 32 and N = 64 and bound both
    errors absolutely. Returns the coarse-mesh error.

    `tol_lo` / `tol_hi` are about 3x the measured coarse / fine errors, so
    a defect that scales the error uniformly (and leaves the rate intact)
    still fails.
    """
    var err_lo = _one_run[P, HLLD](ctx, 32, 0, drho_tol, de_tol)
    var err_hi = _one_run[P, HLLD](ctx, 64, 0, drho_tol, de_tol)
    _ = check_convergence(
        "2D MHD Alfvén wave, P = " + String(P), err_lo, err_hi,
        expected, tol_lo, tol_hi,
    )
    return err_lo


def _rel_diff(a: Real, b: Real) -> Real:
    var d = a - b
    if d < Real(0.0): d = -d
    return d / b


def _check_y_axis(var ctx: DeviceContext, rus_x: Real, hlld_x: Real) raises:
    """The same wave along y (a cyclic relabelling of the components) must
    reproduce the x-run error: it sends the non-trivial states through the
    y-normal faces of the DG core and of both numerical fluxes. `rus_x` /
    `hlld_x` are the P=2, N=32 errors of the x runs from the rate checks."""
    var rus_y = _one_run[2, False](ctx, 32, 1, Real(4.5e-5), Real(1.2e-6))
    var hlld_y = _one_run[2, True](ctx, 32, 1, Real(5.2e-5), Real(1.2e-6))
    print("P=2 N=32  Rusanov x / y:", rus_x, rus_y, "  HLLD x / y:", hlld_x, hlld_y)
    var sym_tol = Real(1.0e-9)
    assert_true(
        _rel_diff(rus_y, rus_x) < sym_tol,
        "Rusanov error differs between propagation axes",
    )
    assert_true(
        _rel_diff(hlld_y, hlld_x) < sym_tol,
        "HLLD error differs between propagation axes",
    )
    assert_true(
        hlld_x < rus_x,
        "HLLD should be less dissipative than Rusanov on a smooth wave",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _ = _check_rate[1, False](ctx, Real(2.0), Real(3.2e-3), Real(8.0e-4),
        Real(1.0e-10), Real(1.0e-10))
    var rus_x = _check_rate[2, False](ctx, Real(3.0), Real(2.5e-5), Real(3.2e-6),
        Real(4.5e-5), Real(1.2e-6))
    _ = _check_rate[3, False](ctx, Real(4.0), Real(2.0e-7), Real(1.2e-8),
        Real(2.2e-6), Real(3.2e-8))
    # HLLD (Miyoshi-Kusano) on the same smooth wave.
    var hlld_x = _check_rate[2, True](ctx, Real(3.0), Real(2.2e-5), Real(2.7e-6),
        Real(5.2e-5), Real(1.2e-6))
    _check_y_axis(ctx, rus_x, hlld_x)
    print("PASSED")
