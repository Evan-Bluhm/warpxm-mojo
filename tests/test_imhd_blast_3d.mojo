# 3D MHD blast wave on a triply-periodic mesh.
#
# IC:
#     ρ(x, y, z, 0) = 1
#     u = 0
#     P(x, y, z, 0) = P_amb + ΔP · exp(−r² / σ²)    (localised pressure pulse)
#     B = (B_0, 0, 0)                                (uniform background)
#     E = P/(γ−1) + ½·|B|²
#
# A 3D analogue of the Euler blast wave with a uniform magnetic field.
# Because the IC has uniform B and no current, ∇·B = 0 is preserved
# exactly under ideal MHD — we don't need GLM cleaning here. As the
# pressure pulse expands, the fluid drags the frozen-in field with it.
#
# Validates that 3D ideal MHD + 3D Moe-Rossmanith limiter survives a
# strong (ΔP/P_amb = 5) 3D shock.

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
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime DELTA_P: Real = 5.0
comptime B0: Real = 1.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 5.0e-4
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
    var bx_0_int = mesh.integrate(q, IDX_BX)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var any_nan = False
    var min_rho: Real = 1.0e30
    var max_rho: Real = 0.0
    def scan(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, MHD_NC],
    ) {mut any_nan, mut min_rho, mut max_rho}:
        var rho = v[IDX_RHO]
        if isnan(rho) or isnan(v[IDX_E]):
            any_nan = True
        if rho < min_rho: min_rho = rho
        if rho > max_rho: max_rho = rho
    mesh.visit_nodes(q, scan)

    var rel_mass_drift = fabs(mesh.integrate(q, IDX_RHO) - mass_0) / mass_0
    var rel_bx_drift = fabs(mesh.integrate(q, IDX_BX) - bx_0_int) / fabs(bx_0_int)
    print("3D MHD blast wave — t =", t)
    print("  rel Δmass:                  ", rel_mass_drift)
    print("  rel Δ(∫B_x dV) (Faraday's):", rel_bx_drift)
    print("  ρ range: [", min_rho, ",", max_rho, "]")

    assert_true(not any_nan, "3D MHD blast wave NaN'd")
    assert_true(rel_mass_drift < Real(1.0e-12), "mass not conserved on periodic mesh")
    # ∫ B dV is conserved exactly by ideal MHD on a periodic domain (no
    # source, divergence-form evolution).
    assert_true(rel_bx_drift < Real(1.0e-12), "∫B_x dV drifted — Faraday violation")
    assert_true(min_rho > Real(0.0), "density went non-positive")
    assert_true(max_rho < Real(10.0), "density blew up — limiter unstable")
    print("PASSED")
