# Temporal order of accuracy of the fixed-step integrators.
#
# Every other accuracy test in the suite is dominated by the spatial error,
# so a wrong Butcher coefficient that kept the method consistent would go
# unnoticed. This test isolates the time integrator by the method of
# lines: the DG semi-discretisation on a fixed mesh is an ODE system
# `dq/dt = L(q)`, and its exact solution is approximated by a reference
# run with a very small step. The error of a run with step `dt` against
# that reference contains no spatial error at all, so halving `dt` must
# reduce it by `2^order`:
#
#   * `SSPRK3_1D.step`            — order 3 (ratio 8)
#   * `DormandPrince45_1D.step`   — order 5 (ratio 32)
#
# The problem is a smooth, genuinely nonlinear 1D Euler flow (density,
# velocity and pressure all perturbed), so the nonlinear order conditions
# are exercised, not just the stability polynomial. It is stopped long
# before a shock can form.
#
# Build / run:
#     pixi run mojo run -I src tests/test_temporal_order.mojo

from std.math import cos, log2, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NX: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime T_FINAL: Real = 0.2
comptime N_COARSE: Int = 32      # dt = 6.25e-3
comptime N_FINE: Int = 64        # dt = 3.125e-3
comptime N_REF: Int = 1024       # DP45 reference, dt = 1.95e-4


def _run[USE_DP45: Bool](var ctx: DeviceContext, n_steps: Int) raises -> List[Real]:
    """Integrate to `T_FINAL` in `n_steps` equal steps; return the state."""
    var mesh = BlockMesh1D[P](ctx, Real(0.0), Real(1.0), NX, periodic=True)
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var rho = Real(1.0) + Real(0.2) * sin(TWO_PI * x)
            var u = Real(0.3) + Real(0.2) * cos(TWO_PI * x)
            var pp = Real(1.0) + Real(0.2) * cos(TWO_PI * x + Real(1.0))
            var b = (e * nodes_per_elem + i) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = rho * u
            host[b + IDX_MY] = Real(0.0)
            host[b + IDX_MZ] = Real(0.0)
            host[b + IDX_E] = pp / (GAMMA - Real(1.0)) + Real(0.5) * rho * u * u
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGEulerSolver1D[P](
        Euler(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )

    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState1D[P, EULER_NC]) raises {imm}:
        pass

    var dt = T_FINAL / Real(n_steps)
    comptime if USE_DP45:
        var stepper = DormandPrince45_1D[P, EULER_NC](ctx, mesh.num_elements)
        for _ in range(n_steps):
            stepper.step(ctx, q, dt, compute_rhs)
    else:
        var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)
        for _ in range(n_steps):
            stepper.step(ctx, q, dt, compute_rhs, no_adjust)
    ctx.synchronize()

    var out = List[Real]()
    with q.buf.map_to_host() as h:
        for i in range(q.buf_len):
            out.append(h[i])
    return out^


def _rms_diff(a: List[Real], b: List[Real]) -> Real:
    var acc: Real = 0.0
    for i in range(len(a)):
        var d = a[i] - b[i]
        acc += d * d
    return (acc / Real(len(a))) ** Real(0.5)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var q_ref = _run[True](ctx, N_REF)

    var rk3_coarse = _rms_diff(_run[False](ctx, N_COARSE), q_ref)
    var rk3_fine = _rms_diff(_run[False](ctx, N_FINE), q_ref)
    var rk3_order = log2(rk3_coarse / rk3_fine)
    print("SSPRK3: err(dt) =", rk3_coarse, " err(dt/2) =", rk3_fine, " order =", rk3_order)

    var dp_coarse = _rms_diff(_run[True](ctx, N_COARSE), q_ref)
    var dp_fine = _rms_diff(_run[True](ctx, N_FINE), q_ref)
    var dp_order = log2(dp_coarse / dp_fine)
    print("DP45  : err(dt) =", dp_coarse, " err(dt/2) =", dp_fine, " order =", dp_order)

    # Measured orders 2.992 and 5.002; errors 1.29e-5 and 3.03e-9.
    assert_true(rk3_order > Real(2.85) and rk3_order < Real(3.15), "SSPRK3 is not third order")
    assert_true(dp_order > Real(4.85) and dp_order < Real(5.15), "DP45 is not fifth order")
    assert_true(rk3_coarse < Real(4.0e-5), "SSPRK3 temporal error above tolerance")
    assert_true(dp_coarse < Real(9.0e-9), "DP45 temporal error above tolerance")
    print("PASSED")
