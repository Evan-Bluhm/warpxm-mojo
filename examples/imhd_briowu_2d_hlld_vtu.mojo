# 2D Brio-Wu MHD shock tube with HLLD numerical flux + multi-frame VTU
# output. Showcases the full strong-MHD pipeline:
#   * `DGImhdSolver[2, P, USE_HLLD=True]` — Miyoshi-Kusano 5-state solver,
#     much sharper shocks than Rusanov.
#   * Moe-Rossmanith limiter for Gibbs control at the strong B_y jump.
#   * 1D physics extruded along a thin y-strip — analytic solution at
#     t = 0.1 has fast rarefactions, slow compound waves, contact, and
#     a slow shock, all visible in the dumped VTU frames.
#
# Per frame the driver prints (∫ρ dV, ∫B_x dV, max|Δρ/Δx|) so the sharpness
# of the rho-jump near the contact can be monitored.
#
# The time step follows the CFL condition (`cfl_max = 0.5`).
#
#     pixi run example imhd_briowu_2d_hlld_vtu
#     # then open output/briowu/briowu.pvd in ParaView

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 4
comptime GAMMA: Real = 2.0
comptime T_FINAL: Real = 0.1
comptime NUM_FRAMES: Int = 11
comptime State = DGState[2, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_COPYOUT, bc_top=BC_COPYOUT,  # v_y != 0 in Brio-Wu: y is an outflow, not a wall
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real) -> Array[Real, MHD_NC]:
        var left = x < Real(0.5)
        var rho: Real = 1.0 if left else 0.125
        var pp: Real = 1.0 if left else 0.1
        var by: Real = 1.0 if left else -1.0
        var bx: Real = 0.75
        var v = Array[Real, MHD_NC](fill=Real(0.0))
        v[IDX_RHO] = rho
        v[IDX_E] = pp / (GAMMA - Real(1.0)) + Real(0.5) * (bx * bx + by * by)
        v[IDX_BX] = bx
        v[IDX_BY] = by
        return v^
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)
    var bx_0 = mesh.integrate(q, IDX_BX)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[2, P, USE_HLLD=True](app)
    var stepper = SSPRK3[2, P, MHD_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[2, P, MHD_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var fw = FrameWriter("briowu")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    var mid_node = State.num_nodes_per_elem // 2
    root_print("frame   time   ∫ρ dV   ∫B_x dV   rel Δmass   rel ΔB_x   max|Δρ/Δx|   CFL dt")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_imhd_frame(fw.next_path(t), mesh, q, GAMMA)

        # Largest jump in ρ between x-neighbouring element centres: a
        # measure of how sharp the captured waves are. Each rank scans its
        # own elements (`mesh.nx` per row) and the maximum is taken over
        # ranks.
        var max_grad: Real = 0.0
        with q.buf.map_to_host() as h:
            for e in range(mesh.num_elements):
                if e % mesh.nx == mesh.nx - 1:
                    continue
                var g = fabs(
                    h[State.index(e + 1, mid_node, IDX_RHO)]
                    - h[State.index(e, mid_node, IDX_RHO)]
                )
                if g > max_grad: max_grad = g
        max_grad = global_max(max_grad)

        var mass = mesh.integrate(q, IDX_RHO)
        var bx_int = mesh.integrate(q, IDX_BX)
        root_print(
            frame, t, mass, bx_int,
            fabs(mass - mass_0) / mass_0, fabs(bx_int - bx_0) / bx_0,
            max_grad, suggest_dt(q),
        )

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt, dt_interval=5,
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/briowu.pvd")
