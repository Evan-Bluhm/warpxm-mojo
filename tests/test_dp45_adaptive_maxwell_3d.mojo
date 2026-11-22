# Adaptive DP45 on a 3D Maxwell plane wave.
#
# Same IC as test_maxwell_3d_plane_wave.mojo (TM wave propagating along
# x̂, period 1 at c = 1, uniform in y / z). Linear hyperbolic problem on
# the 3D 6-component Maxwell state — completes the DP45 adaptive demos
# across the full feature matrix.

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import Maxwell
from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_maxwell_3d import DGMaxwellSolver3D
from warpxm.temporal.dormand_prince_3d import DormandPrince45_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 8
comptime NZ: Int = 8
comptime C_LIGHT: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var s = sin(TWO_PI * x)
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MAXWELL_NC
                            host[b + IDX_EX] = Real(0.0)
                            host[b + IDX_EY] = Real(0.0)
                            host[b + IDX_EZ] = s
                            host[b + IDX_BX] = Real(0.0)
                            host[b + IDX_BY] = -s / C_LIGHT
                            host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGMaxwellSolver3D[P](Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)))
    var stepper = DormandPrince45_3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState3D[P, MAXWELL_NC],
        mut q_out: DGState3D[P, MAXWELL_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = DT_INIT
    var num_steps: Int = 0
    var min_dt = dt
    var max_dt: Real = 0.0
    while t < T_FINAL - Real(1.0e-12):
        var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
        var ret = stepper.step_adaptive(
            ctx,
            q,
            dt_try,
            compute_rhs,
            atol=ATOL,
            rtol=RTOL,
        )
        var dt_used = ret[0]
        var dt_next = ret[1]
        # Controller contract: never step further than asked, and keep the
        # suggested next step inside the [0.2, 5] clip of the step taken.
        assert_true(
            dt_used > Real(0.0)
            and dt_used <= dt_try * (Real(1.0) + Real(1.0e-12)),
            "step taken is longer than the step requested",
        )
        assert_true(
            dt_next >= Real(0.2) * dt_used * (Real(1.0) - Real(1.0e-12))
            and dt_next <= Real(5.0) * dt_used * (Real(1.0) + Real(1.0e-12)),
            "suggested next step outside the [0.2, 5] clip",
        )
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break
    ctx.synchronize()

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var err_Ez: Real = 0.0
    var err_By: Real = 0.0
    var max_other: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var xl = mesh.x_min + Real(i) * mesh.dx
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                                var s = sin(TWO_PI * x)
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MAXWELL_NC
                                var d_Ez = h[b + IDX_EZ] - s
                                var d_By = h[b + IDX_BY] - (-s / C_LIGHT)
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                err_Ez += jw * d_Ez * d_Ez
                                err_By += jw * d_By * d_By
                                for idx in [IDX_EX, IDX_EY, IDX_BX, IDX_BZ]:
                                    var v = h[b + idx]
                                    if v > max_other: max_other = v
                                    if -v > max_other: max_other = -v
    var l2_Ez = err_Ez ** Real(0.5)
    var l2_By = err_By ** Real(0.5)

    print("Adaptive DP45 3D Maxwell (P=2, 16x8x8, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 E_z          :", l2_Ez)
    print("  L2 B_y          :", l2_By)
    print("  max |E_x,E_y,B_x,B_z| :", max_other)

    # SSPRK3 baseline at dt=5e-4 is 2000 steps for t = 1.
    assert_true(
        num_steps > 42 and num_steps < 58,
        "step count outside the band around the measured value",
    )
    assert_true(l2_Ez < Real(2.5e-3), "E_z error worse than SSPRK3 baseline")
    assert_true(l2_By < Real(2.5e-3), "B_y error worse than SSPRK3 baseline")
    # DP45 takes much larger dt than SSPRK3 — floating-point roundoff in the
    # transverse components is amplified accordingly. Still many orders of
    # magnitude below the wave amplitude.
    assert_true(max_other < Real(1.0e-8),
                "transverse components grew — 3D Maxwell coupling spurious")
    print("PASSED")
