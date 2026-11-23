# Generic kinetic kernels at the largest polynomial order each phase
# space allows.
#
# The right-hand-side kernel runs one phase element per GPU block, so
# `(P_x + 1)^XDIM (P_v + 1)^VDIM ≤ 1024` caps the order: 4 (the basis
# limit) through 2D2V, 3 in 2D3V (exactly 1024 threads), 2 in 3D3V. The
# moment kernel has its own, register-bound, block limit. A kernel that
# asks for too much fails at launch, not at compile time, so each
# combination has to be run once: this launches the right-hand side, the
# moments and the BGK source at every limit and checks the density of
# `f = 1`, which is the velocity-space volume to round-off.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_order_limits.mojo

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime V_MAX: Real = 2.0


def launch_all[XDIM: Int, VDIM: Int, P: Int](var ctx: DeviceContext) raises:
    comptime ND = XDIM + VDIM
    comptime State = PhaseState[XDIM, VDIM, P, P, VLASOV_NC]
    var mesh = PhaseMesh[XDIM, VDIM, P, P](
        ctx,
        lo=Array[Real, ND](fill=-V_MAX), hi=Array[Real, ND](fill=V_MAX),
        n=Array[Int, ND](fill=3), periodic=Array[Bool, XDIM](fill=True),
    )
    var em = DGState[XDIM, P, MAXWELL_NC](ctx, mesh.num_x_elements)
    em.buf.enqueue_fill(Real(0.2))
    var q = State(ctx, mesh.num_phase_elements)
    var rhs = State(ctx, mesh.num_phase_elements)
    q.buf.enqueue_fill(Real(1.0))

    DGVlasovSolver[XDIM, VDIM, P, P](VlasovMaxwell()).compute_rhs(
        ctx, mesh, q, em, rhs,
    )
    var mom = DGState[XDIM, P, VLASOV_MOMENT_NC](ctx, mesh.num_x_elements)
    VlasovMoments[XDIM, VDIM, P, P](ctx, mesh).compute(ctx, mesh, q, mom)
    var bgk = VlasovBGKSource[XDIM, VDIM, P, P](ctx, mesh)
    bgk.apply(ctx, mesh, q, rhs, Real(1.0))
    ctx.synchronize()

    var expected = (Real(2.0) * V_MAX) ** VDIM
    var max_err: Real = 0.0
    with mom.buf.map_to_host() as h:
        for node in range(mom.total_nodes):
            max_err = max(
                max_err,
                abs(h[node * VLASOV_MOMENT_NC + MOM_DENSITY] - expected),
            )
    print(
        XDIM, "D", VDIM, "V  P =", P,
        " max |density − volume| =", max_err,
    )
    assert_true(max_err < Real(1.0e-11) * expected, "density of f = 1 is off")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    launch_all[1, 1, 4](ctx)
    launch_all[1, 2, 4](ctx)
    launch_all[1, 3, 4](ctx)
    launch_all[2, 2, 4](ctx)
    launch_all[2, 3, 3](ctx)
    launch_all[3, 3, 2](ctx)
    print("PASSED")
