# 2D Kelvin-Helmholtz instability — shear layer rolls up into vortices.
#
# A canonical hydrodynamic instability test. Two horizontal layers with
# opposite velocities slip past each other; a small perpendicular
# velocity perturbation seeds the rollup. Vortices form on the
# wavelength of the perturbation and merge over time.
#
# Setup (Springel 2010 / Athena-style):
#   * Periodic [0, 1]² mesh.
#   * Two-layer flow with smoothed tanh interfaces at y = 1/4 and 3/4
#     so the IC is C^∞ — avoids Gibbs ringing at the discontinuity that
#     would dominate the L2 error before any KH structure develops:
#       ρ(y) = 1 + Δρ · S(y)
#       u_x(y) = u_inner · S(y) + u_outer · (1 − S(y))
#     where S(y) = ½(tanh((y − 0.25)/σ) − tanh((y − 0.75)/σ)).
#   * Initial v-perturbation:
#       u_y(x, y) = δ_v · sin(4π x) · [exp(−(y − 0.25)²/σ²)
#                                     + exp(−(y − 0.75)²/σ²)]
#   * Uniform pressure P_0 = 2.5 (γ = 5/3).
#
# Multi-frame VTU output. Watch vortices form by frame 4-6.
#
# The time step follows the CFL condition. The flow is subsonic and smooth
# until the billows roll up, so `cfl_max = 1` is used rather than the 0.5
# kept for strong shocks.
#
#     pixi run example euler_kelvin_helmholtz_2d_vtu
#     # then open output/kh/kh.pvd in ParaView

from std.collections import Array
from std.math import exp, tanh, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 128
comptime NY: Int = 128
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO_OUTER: Real = 1.0
comptime RHO_INNER: Real = 2.0
comptime U_OUTER: Real = -0.5
comptime U_INNER: Real = +0.5
comptime PRESSURE: Real = 2.5
comptime SIGMA: Real = 0.025          # tanh / Gaussian width
comptime DELTA_V: Real = 0.01         # v-perturbation amplitude
comptime T_FINAL: Real = 2.0
comptime NUM_FRAMES: Int = 11         # frames 0 .. 10 inclusive
comptime State = DGState[2, P, EULER_NC]


@always_inline
def _shape(y: Real) -> Real:
    """Smooth indicator for the inner layer: ~1 on (0.25, 0.75), ~0 outside."""
    return Real(0.5) * (tanh((y - Real(0.25)) / SIGMA)
                        - tanh((y - Real(0.75)) / SIGMA))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var S = _shape(y)
        var rho = RHO_OUTER + (RHO_INNER - RHO_OUTER) * S
        var ux = U_OUTER + (U_INNER - U_OUTER) * S
        # v-perturbation localised near the two interfaces.
        var bump1 = exp(-(y - Real(0.25)) * (y - Real(0.25)) / (SIGMA * SIGMA))
        var bump2 = exp(-(y - Real(0.75)) * (y - Real(0.75)) / (SIGMA * SIGMA))
        var uy = DELTA_V * sin(Real(2.0) * TWO_PI * x) * (bump1 + bump2)
        var ke = Real(0.5) * rho * (ux * ux + uy * uy)
        return [
            rho, rho * ux, rho * uy, Real(0.0),
            PRESSURE / (GAMMA - Real(1.0)) + ke,
        ]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(1.0),
    )
    var spatial = DGEulerSolver[2, P](app)
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements)
    var fw = FrameWriter("kh")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    var mass_0 = mesh.integrate(q, IDX_RHO)
    var e_0 = mesh.integrate(q, IDX_E)
    print("Kelvin-Helmholtz instability, P=", P, ", N=", NX, "x", NY)
    print("time   ∫ρ dV   ∫E dV   rel Δmass   rel ΔE   CFL dt")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_euler_frame(fw.next_path(t), mesh, q, GAMMA)
        var mass = mesh.integrate(q, IDX_RHO)
        var e_tot = mesh.integrate(q, IDX_E)
        print(
            t, mass, e_tot, (mass - mass_0) / mass_0, (e_tot - e_0) / e_0,
            suggest_dt(q),
        )
        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt, dt_interval=10,
            )

    fw.write_pvd()
    print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/kh.pvd")
