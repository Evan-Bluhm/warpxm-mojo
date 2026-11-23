# Multi-rank regression: 2D Euler blast in a walled box.
#
# Exercises everything a shock problem uses across partition faces: the
# slope limiter (which needs the neighbours' cell means), the positivity
# preserver, wall and copy-out boundary conditions on the ranks that touch
# the boundary, and the CFL step (a global maximum over ranks). With the
# step taken from an exact global maximum, the field is bitwise the same
# for any number of ranks.

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *

comptime P: Int = 2
comptime GAMMA: Real = 1.4
comptime State = DGState[2, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.75),
        nx=30, ny=22, periodic_x=False, periodic_y=False,
        bc_left=BC_WALL_FREESLIP, bc_right=BC_WALL_FREESLIP,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    # Over-pressured disc, off-centre so that nothing is symmetric about a
    # partition face.
    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var r2 = (x - Real(0.4)) * (x - Real(0.4)) + (y - Real(0.3)) * (y - Real(0.3))
        var pp: Real = 10.0 if r2 < Real(0.01) else 0.1
        return [Real(1.0), Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-10),
        min_pressure=Real(1e-10), cfl_max=Real(0.4),
    )
    var spatial = DGEulerSolver[2, P](app)
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var limiter = MoeRossmanith[2, P, EULER_NC](
        ctx, mesh.num_elements, mesh.num_ghost, alpha_coeff=Real(10.0),
    )
    var positivity = PositivityEuler[2, P](
        num_elements=mesh.num_elements, gamma=GAMMA,
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    # Short enough that the shock has reflected off the bottom wall but
    # not reached the open top, so the box is still closed.
    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), Real(0.07), compute_rhs, adjust, suggest_dt,
        dt_interval=2,
    )
    var mass_1 = mesh.integrate(q, IDX_RHO)
    var rho_min: Real = 1.0e30
    def visit(x: Real, y: Real, w: Real, v: Array[Real, EULER_NC]) {mut rho_min}:
        if v[IDX_RHO] < rho_min:
            rho_min = v[IDX_RHO]
    mesh.visit_nodes(q, visit)
    rho_min = global_min(rho_min)
    var sum = mesh.checksum(q)
    if is_root():
        print("ranks =", num_ranks(), " mass drift =", mass_1 - mass_0,
              " min rho =", rho_min)
        print("CHECKSUM", sum)
    if not (fabs(mass_1 - mass_0) < Real(1.0e-12)):
        raise Error("mass not conserved in the closed box")
    if not (rho_min > Real(0.0)):
        raise Error("density lost positivity")
    if is_root():
        print("PASSED")
