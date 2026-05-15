# 1D scalar advection on a periodic block mesh.
#
# Mirrors `examples/dg/advection/advection.py` from upstream WARPXM, restricted
# to one dimension and to a periodic domain with SSPRK3 time stepping. The
# initial condition is a single sine period on x ∈ [-0.5, 0.5]; with velocity
# v = 1 the exact solution returns to the IC at t = 1.
#
# Build / run:
#     pixi run mojo run -I src examples/advection_1d.mojo
#
# This is a compiled input file — there is no runtime input parsing. Edit the
# constants near the top to change the run.

from std.math import ceildiv, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.apps.advection import Advection1D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2                  # polynomial order (3 GLL nodes / elem)
comptime NUM_ELEMENTS: Int = 64
comptime X_MIN: Real = -0.5
comptime X_MAX: Real = 0.5
comptime VELOCITY: Real = 1.0
comptime T_FINAL: Real = 1.0         # one full period
comptime DT: Real = 1.0e-3
comptime NUM_PRINT_NODES: Int = 8    # how many nodes to dump at start/end


def init_sine(var ctx: DeviceContext, mut q: DGState1D[P, 1], mesh: BlockMesh1D[P]) raises:
    """Fill `q` with q(x) = sin(2π x) on the mesh's node coordinates."""
    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.total_nodes)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var xi = ref_nodes[i]
            var x = xl + (Real(1.0) + xi) * mesh.jacobian
            host[e * nodes_per_elem + i] = sin(TWO_PI * x)

    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)


def dump_first_few(label: String, var ctx: DeviceContext, q: DGState1D[P, 1]) raises:
    """Map q to host and print the first NUM_PRINT_NODES values."""
    with q.buf.map_to_host() as h:
        print(label)
        var n = NUM_PRINT_NODES if NUM_PRINT_NODES < q.total_nodes else q.total_nodes
        for k in range(n):
            print("  q[", k, "] =", h[k])


def l2_error(var ctx: DeviceContext, q: DGState1D[P, 1], mesh: BlockMesh1D[P]) raises -> Real:
    """L2 error of q against the IC sin(2π x). Useful as a sanity metric after
    advancing one full period on a periodic domain — the answer should be ≪ 1.
    """
    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()

    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var xi = ref_nodes[i]
                var x = xl + (Real(1.0) + xi) * mesh.jacobian
                var exact = sin(TWO_PI * x)
                var num = h[e * nodes_per_elem + i]
                var d = num - exact
                acc += weights[i] * mesh.jacobian * d * d
    return acc ** Real(0.5)


def main() raises:
    comptime assert has_accelerator(), "warpxm-mojo requires a GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](ctx, X_MIN, X_MAX, NUM_ELEMENTS, periodic=True)
    var q = DGState1D[P, 1](ctx, mesh.num_elements)
    init_sine(ctx, q, mesh)
    dump_first_few("Initial q (first nodes):", ctx, q)

    var app = Advection1D[1](velocity_x=VELOCITY, cfl_max=Real(1.0))
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
    var step: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
        step += 1

    ctx.synchronize()
    print("\nIntegrated to t =", t, "in", step, "steps")
    dump_first_few("Final q (first nodes):", ctx, q)
    var err = l2_error(ctx, q, mesh)
    print("L2 error vs IC after one period:", err)
