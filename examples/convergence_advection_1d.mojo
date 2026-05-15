# Convergence study for 1D scalar advection.
#
# Advects sin(2π x) for one period on a periodic mesh and reports L2 error
# vs the exact (initial) solution for each (P, N) combination. With SSPRK3
# time stepping at small Δt the spatial error dominates, so the L2 error
# should scale like dx^(P + 1) for smooth solutions.
#
# Build / run:
#     pixi run mojo run -I src examples/convergence_advection_1d.mojo

from std.math import log2, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.apps.advection import Advection1D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


def run_once[P: Int](
    var ctx: DeviceContext, num_elements: Int, dt: Real, t_final: Real,
) raises -> Real:
    """Run advection at order P with N elements and return the post-period L2 error."""
    var mesh = BlockMesh1D[P](ctx, Real(-0.5), Real(0.5), num_elements, periodic=True)
    var q = DGState1D[P, 1](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()

    # IC: sin(2π x)
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.total_nodes)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            host[e * nodes_per_elem + i] = sin(TWO_PI * x)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Advection1D[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    var spatial = DGSpatialSolver1D[P, 1](app)
    var stepper = SSPRK3_1D[P, 1](ctx, mesh.num_elements)


    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, 1],
        mut q_out: DGState1D[P, 1],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState1D[P, 1]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var exact = sin(TWO_PI * x)
                var d = h[e * nodes_per_elem + i] - exact
                acc += weights[i] * mesh.jacobian * d * d
    return acc ** Real(0.5)


def sweep_p[P: Int](var ctx: DeviceContext) raises:
    """Run two resolutions at order P and print observed convergence rate."""
    # Pick dt so the time-integration error stays below the spatial error.
    # SSPRK3 is third-order in time; with dt = 5e-4 over one period we contribute
    # ~ (5e-4)^3 · O(1) ≈ 1e-10, well below the spatial errors below.
    var dt: Real = 5.0e-4
    var t_final: Real = 1.0

    var err_lo = run_once[P](ctx, 32, dt, t_final)
    var err_hi = run_once[P](ctx, 64, dt, t_final)
    var rate = log2(err_lo / err_hi)
    print(
        "  P =", P,
        " N = 32 ⇒ L2 =", err_lo,
        " | N = 64 ⇒ L2 =", err_hi,
        " | rate ≈", rate,
        " (expected ≈", P + 1, ")",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("1D advection convergence — sin(2π x) on [-0.5, 0.5], one period")
    sweep_p[1](ctx)
    sweep_p[2](ctx)
    sweep_p[3](ctx)
    sweep_p[4](ctx)
