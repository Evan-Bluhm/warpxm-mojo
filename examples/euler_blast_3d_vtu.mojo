# 3D Euler blast wave with multi-frame VTU output.
#
# A Gaussian pressure pulse at the center of a triply-periodic box drives
# a spherical shock wave outward. Demonstrates the 3D pipeline end-to-end:
#   * BlockMesh[3, P] (periodic 3-axis)
#   * Euler with γ-law EOS
#   * DGEulerSolver[3, P] + MoeRossmanith[3, P, EULER_NC] for shock capturing
#   * SSPRK3 with a CFL-driven time step (`run_until_cfl`)
#   * FrameWriter + dump_euler_frame for ParaView animation
#
# Per frame the driver prints (∫ρ dV, ∫E dV, max ρ, min ρ) so the user
# can watch conservation hold and the rarefaction at the centre grow.
#
#     pixi run example euler_blast_3d_vtu
#     # then open output/euler_blast_3d/blast.pvd in ParaView

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime DELTA_P: Real = 5.0
comptime T_FINAL: Real = 0.15
comptime NUM_FRAMES: Int = 8
comptime State = DGState[3, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, EULER_NC]:
        var r2 = (
            (x - Real(0.5)) * (x - Real(0.5)) + (y - Real(0.5)) * (y - Real(0.5))
            + (z - Real(0.5)) * (z - Real(0.5))
        )
        var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
        return [Real(1.0), Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[3, P](app)
    var stepper = SSPRK3[3, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[3, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var fw = FrameWriter("blast", "output/euler_blast_3d")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    root_print("frame   time   ∫ρ dV   ∫E dV   min ρ   max ρ   CFL dt")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_euler_frame(fw.next_path(t), mesh, q, GAMMA)
        var min_rho: Real = 1.0e30
        var max_rho: Real = 0.0
        def extrema(
            x: Real, y: Real, z: Real, w: Real, v: Array[Real, EULER_NC],
        ) {mut min_rho, mut max_rho}:
            min_rho = fmin(min_rho, v[IDX_RHO])
            max_rho = fmax(max_rho, v[IDX_RHO])
        mesh.visit_nodes(q, extrema)
        # `visit_nodes` covers this rank's elements; combine across ranks.
        min_rho = global_min(min_rho)
        max_rho = global_max(max_rho)
        root_print(
            frame, t, mesh.integrate(q, IDX_RHO), mesh.integrate(q, IDX_E),
            min_rho, max_rho, suggest_dt(q),
        )

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt, dt_interval=5,
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/blast.pvd")
