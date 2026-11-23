# 2D Sedov-like blast with positivity preservation + multi-frame VTU output.
#
# This driver exercises the full strong-shock pipeline:
#   * `SSPRK3` — the generic three-stage Shu-Osher stepper, with a
#     user-supplied `adjust` closure that chains the Moe-Rossmanith
#     limiter and the Zhang-Shu positivity preserver between stages.
#   * `run_until_cfl` — the time step follows the CFL condition, which
#     matters here: the signal speed drops by a large factor as the blast
#     expands.
#   * VTU output every 0.01 time units so the user can watch the
#     spherically symmetric shock expand in ParaView.
#
# IC: ambient ρ = 1, P_amb = 1e-3 fluid at rest, with a small Gaussian
# pressure pulse at the centre (peak P_inner = 10, so the pressure
# ratio is 10⁴). Without the positivity preserver this NaN's quickly
# as the shock fronts pull P_gas through zero in the rarefaction wake.
#
# The driver prints, per frame, (∫ρ dV, ∫E dV, max |u|, min ρ, min P) and
# the current CFL step — mass and total energy are exactly conserved on
# the periodic domain, while min ρ and min P stay positive.
#
#     pixi run example euler_sedov_blast_2d_vtu
#     # then open output/sedov/sedov.pvd in ParaView

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 64
comptime GAMMA: Real = 1.4
comptime SIGMA: Real = 0.03
comptime P_AMB: Real = 1.0e-3
comptime P_INNER: Real = 10.0           # 10⁴ × ambient
comptime T_FINAL: Real = 0.10
comptime NUM_FRAMES: Int = 11           # frames 0..10
comptime State = DGState[2, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var r2 = (x - Real(0.5)) * (x - Real(0.5)) + (y - Real(0.5)) * (y - Real(0.5))
        var pp = P_AMB + (P_INNER - P_AMB) * exp(-r2 / (SIGMA * SIGMA))
        return [Real(1.0), Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)
    var e_0 = mesh.integrate(q, IDX_E)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[2, P](app)
    # The limiter acts on deviations above α = alpha_coeff · dx^1.5. The
    # default coefficient (1000) gives α ≈ 2 here, above everything in the
    # problem, so the limiter would be idle and the rarefied core would be
    # held up by the positivity floor alone. With a CFL-driven step that
    # turns into a collapse: an evacuating node raises the sound speed,
    # which shrinks dt, which lets it evacuate further. α ≈ 0.2 keeps the
    # limiter working on the blast without flattening it.
    var limiter = MoeRossmanith[2, P, EULER_NC](
        ctx, mesh.num_elements, mesh.num_ghost, alpha_coeff=Real(100.0),
    )
    var positivity = PositivityEuler[2, P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var fw = FrameWriter("sedov")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    root_print("frame   time   ∫ρ dV   ∫E dV   max|u|   min ρ   min P_gas   rel Δmass   rel ΔE   CFL dt")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_euler_frame(fw.next_path(t), mesh, q, GAMMA)

        var max_u: Real = 0.0
        var min_rho: Real = 1.0e30
        var min_p: Real = 1.0e30
        def extrema(
            x: Real, y: Real, w: Real, v: Array[Real, EULER_NC],
        ) {mut max_u, mut min_rho, mut min_p}:
            var rho = v[IDX_RHO]
            var ux = v[IDX_MX] / rho
            var uy = v[IDX_MY] / rho
            var ke = Real(0.5) * rho * (ux * ux + uy * uy)
            max_u = fmax(max_u, sqrt(ux * ux + uy * uy))
            min_rho = fmin(min_rho, rho)
            min_p = fmin(min_p, (GAMMA - Real(1.0)) * (v[IDX_E] - ke))
        mesh.visit_nodes(q, extrema)
        # `visit_nodes` covers this rank's elements; combine across ranks.
        max_u = global_max(max_u)
        min_rho = global_min(min_rho)
        min_p = global_min(min_p)
        var mass = mesh.integrate(q, IDX_RHO)
        var e_int = mesh.integrate(q, IDX_E)
        root_print(
            frame, t, mass, e_int, max_u, min_rho, min_p,
            fabs(mass - mass_0) / mass_0, fabs(e_int - e_0) / e_0,
            suggest_dt(q),
        )

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt,
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/sedov.pvd")
