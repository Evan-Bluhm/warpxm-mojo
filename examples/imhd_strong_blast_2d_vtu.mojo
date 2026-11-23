# 2D strong MHD blast — ΔP/P_amb = 1000 spherical pulse in a B₀ = 5
# background — with multi-frame VTU output. Demonstrates the
# `PositivityMHD` adjuster in action: without it the simulation
# would NaN almost immediately as the shock fronts drive P_gas
# through zero in the post-shock rarefaction wake.
#
# Per frame the driver prints (min ρ, min P_gas, ∫ρ dV, ∫B_x dV,
# ∫E dV) — mass and ∫B_x are exactly conserved on the periodic
# domain, while min ρ and min P_gas remain positive throughout.
#
# The time step follows the CFL condition (`cfl_max = 0.5`).
#
#     pixi run example imhd_strong_blast_2d_vtu
#     # then open output/mhd_strong_blast_2d/blast.pvd in ParaView

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 64
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.05
comptime P_AMB: Real = 0.1
comptime DELTA_P: Real = 100.0
comptime B0: Real = 5.0
comptime T_FINAL: Real = 0.02
comptime NUM_FRAMES: Int = 11
comptime State = DGState[2, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real) -> Array[Real, MHD_NC]:
        var r2 = (x - Real(0.5)) * (x - Real(0.5)) + (y - Real(0.5)) * (y - Real(0.5))
        var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
        var v = Array[Real, MHD_NC](fill=Real(0.0))
        v[IDX_RHO] = 1.0
        v[IDX_E] = pp / (GAMMA - Real(1.0)) + Real(0.5) * B0 * B0
        v[IDX_BX] = B0
        return v^
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)
    var bx_0 = mesh.integrate(q, IDX_BX)
    var e_0 = mesh.integrate(q, IDX_E)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[2, P](app)
    var stepper = SSPRK3[2, P, MHD_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[2, P, MHD_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var positivity = PositivityMHD[2, P, MHD_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var fw = FrameWriter("blast", "output/mhd_strong_blast_2d")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    root_print("frame   time   min ρ   min P_gas   ∫ρ dV   ∫B_x dV   ∫E dV   relΔm   relΔB_x   relΔE   CFL dt")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_imhd_frame(fw.next_path(t), mesh, q, GAMMA)

        var min_rho: Real = 1.0e30
        var min_p: Real = 1.0e30
        def extrema(
            x: Real, y: Real, w: Real, v: Array[Real, MHD_NC],
        ) {mut min_rho, mut min_p}:
            var rho = v[IDX_RHO]
            var ke = Real(0.5) * (
                v[IDX_MX] * v[IDX_MX] + v[IDX_MY] * v[IDX_MY] + v[IDX_MZ] * v[IDX_MZ]
            ) / rho
            var mag = Real(0.5) * (
                v[IDX_BX] * v[IDX_BX] + v[IDX_BY] * v[IDX_BY] + v[IDX_BZ] * v[IDX_BZ]
            )
            min_rho = fmin(min_rho, rho)
            min_p = fmin(min_p, (GAMMA - Real(1.0)) * (v[IDX_E] - ke - mag))
        mesh.visit_nodes(q, extrema)
        # `visit_nodes` covers this rank's elements; combine across ranks.
        min_rho = global_min(min_rho)
        min_p = global_min(min_p)
        var mass = mesh.integrate(q, IDX_RHO)
        var bx_int = mesh.integrate(q, IDX_BX)
        var e_int = mesh.integrate(q, IDX_E)
        root_print(
            frame, t, min_rho, min_p, mass, bx_int, e_int,
            fabs(mass - mass_0) / mass_0, fabs(bx_int - bx_0) / bx_0,
            fabs(e_int - e_0) / e_0, suggest_dt(q),
        )

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt,
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/blast.pvd")
