# Einfeldt 1-2-3 problem — classic near-vacuum stress test for positivity
# preservation in Euler. Two strong outward rarefactions create a region
# where ρ and P collapse toward zero. The conserved-variable limiter
# alone cannot prevent negative density or pressure here; the
# Zhang-Shu positivity preserver in `adjusters/positivity_euler_1d.mojo`
# is what keeps the simulation valid.
#
# IC (Einfeldt et al. 1991):
#   Left  (x < 0.5): ρ = 1, u = -2, P = 0.4
#   Right (x ≥ 0.5): ρ = 1, u = +2, P = 0.4
#   γ = 1.4, t_final = 0.15
#
# Expected (analytic): minimum density ≈ 0.022, minimum pressure ≈ 0.003
# at the symmetry centre. Anything > 0 at t = 0.15 demonstrates the
# preserver is doing its job.

from std.math import isnan
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith_1d import MoeRossmanith1D
from warpxm.adjusters.positivity_euler_1d import PositivityEuler1D
from warpxm.apps.euler_1d import Euler1D
from warpxm.apps.euler_2d import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NX: Int = 200
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.15
comptime DT: Real = 1.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), num_elements=NX, periodic=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for i in range(NX):
        var xl = mesh.x_min + Real(i) * mesh.dx
        for in_ in range(P1):
            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian
            var rho: Real = 1.0
            var u: Real
            var pp: Real = 0.4
            if x < Real(0.5):
                u = -2.0
            else:
                u = +2.0
            var ke = Real(0.5) * rho * u * u
            var E_tot = pp / (GAMMA - Real(1.0)) + ke
            var b = (i * P1 + in_) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = rho * u
            host[b + IDX_MY] = Real(0.0)
            host[b + IDX_MZ] = Real(0.0)
            host[b + IDX_E] = E_tot
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler1D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver1D[P](app)
    var limiter = MoeRossmanith1D[P](ctx, mesh.num_elements)
    var positivity = PositivityEuler1D[P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)


    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState1D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var any_nan = False
    var min_rho: Real = 1.0e30
    var min_p: Real = 1.0e30
    var max_rho: Real = -1.0e30
    var center_rho: Real = 0.0
    var center_p: Real = 0.0
    with q.buf.map_to_host() as h:
        for i in range(NX):
            for in_ in range(P1):
                var b = (i * P1 + in_) * EULER_NC
                var rho = h[b + IDX_RHO]
                var mx = h[b + IDX_MX]
                var E = h[b + IDX_E]
                if isnan(rho) or isnan(E): any_nan = True
                var u = mx / rho
                var ke = Real(0.5) * rho * u * u
                var pp = (GAMMA - Real(1.0)) * (E - ke)
                if rho < min_rho: min_rho = rho
                if rho > max_rho: max_rho = rho
                if pp < min_p: min_p = pp
                if i == NX // 2 and in_ == 0:
                    center_rho = rho
                    center_p = pp

    print("1D Einfeldt 1-2-3 problem — t =", t)
    print("  min ρ:                  ", min_rho)
    print("  max ρ:                  ", max_rho)
    print("  min P:                  ", min_p)
    print("  ρ at center:            ", center_rho)
    print("  P at center:            ", center_p)

    assert_true(not any_nan, "1-2-3 problem NaN'd — positivity preserver failed")
    assert_true(min_rho > Real(0.0),
                "negative density survived — preserver insufficient")
    assert_true(min_p > Real(0.0),
                "negative pressure survived — preserver insufficient")
    print("PASSED")
