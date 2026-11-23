# Weak-scaling bench: 3D GLM-MHD with the limiter, one N³ block per rank.
#
# The global mesh grows with the number of ranks so that every rank owns
# the same N × N × N block, and the mesh is periodic, so every rank
# exchanges all six faces. Ideal weak scaling is a step rate that does not
# depend on the number of ranks.
#
#     pixi run mpi-build bench/bench_mpi_weak_imhd_glm_3d.mojo
#     mpirun -np 8 build/mpi/bench_mpi_weak_imhd_glm_3d     # one GPU per rank
#
# The block size is the define `N` (default 32):
#
#     pixi run mpi-build bench/bench_mpi_weak_imhd_glm_3d.mojo -D N=96
#
# Ranks that share a GPU take turns on it, so the step rate only means
# something with one GPU per rank.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *
from warpxm.parallel.partition import process_grid


comptime P: Int = 2
comptime N: Int = get_defined_int["N", 32]()
comptime GAMMA: Real = 5.0 / 3.0
comptime C_H: Real = 4.0
comptime C_P_SQ: Real = 0.5
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100
comptime State = DGState[3, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()

    # As cubic a process grid as the rank count allows, N elements per box.
    var ranks = num_ranks()
    var procs = process_grid([ranks, ranks, ranks], 3, ranks)
    var mesh = BlockMesh[3, P](
        ctx,
        Real(0.0), TWO_PI * Real(procs[0]),
        Real(0.0), TWO_PI * Real(procs[1]),
        Real(0.0), TWO_PI * Real(procs[2]),
        nx=N * procs[0], ny=N * procs[1], nz=N * procs[2],
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    if mesh.num_elements != N * N * N:
        raise Error("the mesh was not cut into N^3 blocks")
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    var rho_0 = GAMMA * GAMMA
    var p_0: Real = GAMMA
    # 2D Orszag-Tang-like IC extruded along z, one period per box.
    def initial(x: Real, y: Real, z: Real) {imm} -> Array[Real, MHD_GLM_NC]:
        var ux = -sin(y)
        var uy = sin(x)
        var bx = -sin(y)
        var by = sin(Real(2.0) * x)
        var e_tot = (
            p_0 / (GAMMA - Real(1.0))
            + Real(0.5) * rho_0 * (ux * ux + uy * uy)
            + Real(0.5) * (bx * bx + by * by)
        )
        return [
            rho_0, rho_0 * ux, rho_0 * uy, Real(0.0), e_tot,
            bx, by, Real(0.0), Real(0.0),
        ]
    mesh.fill(q, initial)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_GLM_NC](
        ctx, mesh.num_elements, mesh.num_ghost,
    )
    var limiter = MoeRossmanith[3, P, MHD_GLM_NC](
        ctx, mesh.num_elements, mesh.num_ghost,
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    # A CFL step for this resolution, the same on every rank.
    var dt = spatial.suggest_dt(ctx, mesh, q)

    for _ in range(WARMUP):
        stepper.step(ctx, q, dt, compute_rhs, adjust)
    ctx.synchronize()
    _ = global_sum(Real(0.0))   # all ranks start the clock together

    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step(ctx, q, dt, compute_rhs, adjust)
    ctx.synchronize()
    _ = global_sum(Real(0.0))
    var t1 = perf_counter_ns()

    var elapsed_s = Real(t1 - t0) / Real(1.0e9)
    var steps_per_s = Real(N_STEPS) / elapsed_s
    var dof_per_rank = Real(mesh.num_elements * State.num_nodes_per_elem)
    root_print(
        "BENCH config=mpi_weak_imhd_glm_3d ranks=" + String(ranks)
        + " procs=" + String(procs[0]) + "x" + String(procs[1]) + "x"
        + String(procs[2])
        + " block=" + String(N) + "^3"
        + " ghost_elems=" + String(mesh.num_ghost)
        + " gpu_aware=" + String(mesh.halo.gpu_aware)
        + " steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
        + " node_updates_per_s_per_rank=" + String(dof_per_rank * steps_per_s)
    )
