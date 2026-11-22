# 3D strong MHD blast — 3D analog of `test_imhd_strong_blast_2d.mojo`.
# Spherical pressure pulse (ΔP/P_amb = 100, σ = 0.05) in a strong B₀ = 5
# magnetic background. Without positivity preservation the simulation
# NaNs as the shock fronts drive P_gas through zero.

from std.collections import Array
from std.math import exp, isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.05
comptime P_AMB: Real = 0.1
comptime DELTA_P: Real = 100.0
comptime B0: Real = 5.0
comptime T_FINAL: Real = 0.02
comptime DT: Real = 2.5e-4
comptime State = DGState[3, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, MHD_NC]:
        var r2 = (x - Real(0.5)) * (x - Real(0.5))
        r2 += (y - Real(0.5)) * (y - Real(0.5))
        r2 += (z - Real(0.5)) * (z - Real(0.5))
        var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
        var v = Array[Real, MHD_NC](fill=Real(0.0))
        v[IDX_RHO] = 1.0
        v[IDX_E] = pp / (GAMMA - Real(1.0)) + Real(0.5) * B0 * B0
        v[IDX_BX] = B0
        return v^
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)
    var bx_0 = mesh.integrate(q, IDX_BX)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, MHD_NC](ctx, mesh.num_elements)
    var positivity = PositivityMHD[3, P, MHD_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var any_nan = False
    var min_rho: Real = 1.0e30
    var max_rho: Real = -1.0e30
    var min_p: Real = 1.0e30
    def scan(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, MHD_NC],
    ) {mut any_nan, mut min_rho, mut max_rho, mut min_p}:
        var rho = v[IDX_RHO]
        if isnan(rho) or isnan(v[IDX_E]): any_nan = True
        var ke = Real(0.5) * (
            v[IDX_MX] * v[IDX_MX] + v[IDX_MY] * v[IDX_MY] + v[IDX_MZ] * v[IDX_MZ]
        ) / rho
        var mag = Real(0.5) * (
            v[IDX_BX] * v[IDX_BX] + v[IDX_BY] * v[IDX_BY] + v[IDX_BZ] * v[IDX_BZ]
        )
        var pp = (GAMMA - Real(1.0)) * (v[IDX_E] - ke - mag)
        if rho < min_rho: min_rho = rho
        if rho > max_rho: max_rho = rho
        if pp < min_p: min_p = pp
    mesh.visit_nodes(q, scan)
    var mass_T = mesh.integrate(q, IDX_RHO)
    var bx_T = mesh.integrate(q, IDX_BX)

    print("3D strong MHD blast — t =", t)
    print("  min ρ:                    ", min_rho)
    print("  min P_gas:                ", min_p)
    print("  rel Δmass:                ", fabs(mass_T - mass_0) / mass_0)
    print("  rel Δ(∫B_x dV):           ", fabs(bx_T - bx_0) / bx_0)

    assert_true(not any_nan, "3D strong MHD blast NaN'd")
    assert_true(fabs(mass_T - mass_0) / mass_0 < Real(1.0e-11),
                "mass not conserved (limiter / positivity must be conservative)")
    assert_true(min_rho > Real(0.0), "negative ρ survived preserver")
    assert_true(min_p > Real(0.0), "negative P_gas survived preserver")
    assert_true(fabs(bx_T - bx_0) / bx_0 < Real(1.0e-10),
                "∫B_x not preserved")
    print("PASSED")
