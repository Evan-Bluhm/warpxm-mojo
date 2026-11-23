# 3D MHD blast wave with multi-frame VTU output.
#
# Spherical pressure pulse in a uniform magnetic background — the fluid
# drags the frozen-in field as the shock expands. Because the IC has
# uniform B and zero current, ∇·B = 0 is preserved exactly under ideal
# MHD without needing GLM cleaning.
#
# Per frame the driver prints (∫ρ dV, ∫B_x dV, ∫E dV) and their drifts —
# mass and ∫B_x are exactly conserved (periodic + Faraday's law), so any
# drift indicates numerical leakage.
#
# The time step follows the CFL condition (`cfl_max = 0.5`).
#
#     pixi run example imhd_blast_3d_vtu
#     # then open output/imhd_blast_3d/blast.pvd in ParaView

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

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
comptime NUM_FRAMES: Int = 11
comptime State = DGState[3, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, MHD_NC]:
        var r2 = (
            (x - Real(0.5)) * (x - Real(0.5)) + (y - Real(0.5)) * (y - Real(0.5))
            + (z - Real(0.5)) * (z - Real(0.5))
        )
        var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
        var v = Array[Real, MHD_NC](fill=Real(0.0))
        v[IDX_RHO] = 1.0
        v[IDX_E] = pp / (GAMMA - Real(1.0)) + Real(0.5) * B0 * B0
        v[IDX_BX] = B0
        return v^
    mesh.fill(q, initial)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[3, P, MHD_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var fw = FrameWriter("blast", "output/imhd_blast_3d")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    var mass_0 = mesh.integrate(q, IDX_RHO)
    var bx_0 = mesh.integrate(q, IDX_BX)
    var e_0 = mesh.integrate(q, IDX_E)
    root_print("frame   time   ∫ρ dV   ∫B_x dV   ∫E dV   rel Δmass   rel ΔB_x   rel ΔE   CFL dt")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_imhd_frame(fw.next_path(t), mesh, q, GAMMA)
        var mass = mesh.integrate(q, IDX_RHO)
        var bx_int = mesh.integrate(q, IDX_BX)
        var e_int = mesh.integrate(q, IDX_E)
        root_print(
            frame, t, mass, bx_int, e_int,
            (mass - mass_0) / mass_0, (bx_int - bx_0) / bx_0,
            (e_int - e_0) / e_0, suggest_dt(q),
        )

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt, dt_interval=5,
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/blast.pvd")
