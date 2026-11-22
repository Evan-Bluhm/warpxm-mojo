# 1D Sod shock tube — canonical CFD test problem.
#
# Initial discontinuity at x = 0.5 separates two uniform states:
#
#     Left  (x < 0.5):  ρ = 1.0,     u = 0,  P = 1.0
#     Right (x ≥ 0.5):  ρ = 0.125,   u = 0,  P = 0.1
#
# γ = 1.4 (diatomic). The exact Riemann solution (Toro Ch. 4) has, from left
# to right:
#
#     1) Left-moving rarefaction fan
#     2) Constant left state
#     3) Contact discontinuity (jump in ρ, continuous P and u)
#     4) Constant right state
#     5) Right-moving shock
#
# The Moe-Rossmanith slope limiter (conserved-variable cut) damps Gibbs
# oscillations near the discontinuities, and the time step follows the CFL
# condition. The post-shock plateau, the contact, and the rarefaction
# structure should appear at the correct locations at `t = 0.2`.
#
#     pixi run example euler_sod_1d

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 200
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime State = DGState[1, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](
        ctx,
        Real(0.0),
        Real(1.0),
        NUM_ELEMENTS,
        periodic=False,
        bc_left=BC_COPYOUT,
        bc_right=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real) -> Array[Real, EULER_NC]:
        var rho: Real = 1.0 if x < Real(0.5) else 0.125
        var pp: Real = 1.0 if x < Real(0.5) else 0.1
        return [rho, Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[1, P](app)
    var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[1, P, EULER_NC](ctx, mesh.num_elements)

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

    # Print rho, u, P at element centroids (the P = 2 mid node) for a quick
    # text view.
    print("# Sod shock tube at t =", t, " final CFL dt =", suggest_dt(q))
    print("# x, rho, u, P")
    var mid_node = State.num_nodes_per_elem // 2
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var x = mesh.x_min + (Real(e) + Real(0.5)) * mesh.dx
            var rho = h[State.index(e, mid_node, IDX_RHO)]
            var u = h[State.index(e, mid_node, IDX_MX)] / rho
            var ke = Real(0.5) * rho * u * u
            var p_local = (GAMMA - Real(1.0)) * (h[State.index(e, mid_node, IDX_E)] - ke)
            print(x, ",", rho, ",", u, ",", p_local)
