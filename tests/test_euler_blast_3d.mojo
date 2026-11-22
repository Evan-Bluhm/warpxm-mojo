# 3D Euler blast wave on a triply-periodic mesh.
#
# IC:
#     ρ(x, y, z, 0) = 1
#     u = 0
#     P(x, y, z, 0) = P_amb + ΔP · exp(−r² / σ²)
#     E = P / (γ − 1)
#
# with `r = |x − x_c|` and the perturbation centred at the domain centre.
# The pressure pulse drives an outgoing spherical shock; the negative-density
# / negative-pressure failure mode the limiter is designed to suppress
# triggers immediately for ΔP/P_amb ≳ 5 without it.
#
# We check at moderate t = 0.1 (well past shock formation):
#   * no NaN
#   * mass conservation to machine precision (periodic, no sources)
#   * density stays positive
#   * fields remain bounded

from std.collections import Array
from std.math import exp, isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime DELTA_P: Real = 5.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 5.0e-4
comptime State = DGState[3, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, EULER_NC]:
        var r2 = (x - Real(0.5)) * (x - Real(0.5))
        r2 += (y - Real(0.5)) * (y - Real(0.5))
        r2 += (z - Real(0.5)) * (z - Real(0.5))
        var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
        return [Real(1.0), Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[3, P](app)
    var stepper = SSPRK3[3, P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var any_nan = False
    var mass_T: Real = 0.0
    var min_rho: Real = 1.0e30
    var max_rho: Real = 0.0
    def scan(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, EULER_NC],
    ) {mut any_nan, mut mass_T, mut min_rho, mut max_rho}:
        var rho = v[IDX_RHO]
        if isnan(rho) or isnan(v[IDX_E]):
            any_nan = True
        mass_T += w * rho
        if rho < min_rho: min_rho = rho
        if rho > max_rho: max_rho = rho
    mesh.visit_nodes(q, scan)

    var rel_mass_drift = fabs(mass_T - mass_0) / mass_0
    print("3D blast wave — t =", t)
    print("  initial mass:  ", mass_0)
    print("  final mass:    ", mass_T)
    print("  rel Δmass:      ", rel_mass_drift)
    print("  ρ range: [", min_rho, ",", max_rho, "]")
    assert_true(not any_nan, "3D blast wave NaN'd")
    assert_true(rel_mass_drift < Real(1.0e-12), "mass not conserved on periodic mesh")
    assert_true(min_rho > Real(0.0), "density went non-positive")
    assert_true(max_rho < Real(10.0), "density blew up — limiter unstable")
    print("PASSED")
