# 2D Sod shock tube with VTU output.
#
# Same setup as `tests/test_euler_sod_2d.mojo`, with a CFL-driven time step,
# and dumps a `.vtu` snapshot of the t = 0.2 state to `output/sod_2d.vtu` so
# the result can be visualised in ParaView.
#
#     pixi run example euler_sod_2d_vtu

from std.collections import Array
from std.os import makedirs
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 8
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime OUTPUT_DIR: String = "output"
comptime State = DGState[2, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var rho: Real = 1.0 if x < Real(0.5) else 0.125
        var pp: Real = 1.0 if x < Real(0.5) else 0.1
        return [rho, Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[2, P](app)
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    var t = stepper.run_until_cfl(
        ctx, q, Real(0.0), T_FINAL, compute_rhs, adjust, suggest_dt,
        dt_interval=5,
    )

    makedirs(OUTPUT_DIR, exist_ok=True)
    var path = OUTPUT_DIR + "/sod_2d.vtu"
    dump_euler_frame(path, mesh, q, GAMMA)
    root_print("t =", t, " final CFL dt =", suggest_dt(q), " ∫ρ dV =", mesh.integrate(q, IDX_RHO))
    root_print("Wrote", path)
