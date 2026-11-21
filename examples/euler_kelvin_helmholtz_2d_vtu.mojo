# 2D Kelvin-Helmholtz instability — shear layer rolls up into vortices.
#
# A canonical hydrodynamic instability test. Two horizontal layers with
# opposite velocities slip past each other; a small perpendicular
# velocity perturbation seeds the rollup. Vortices form on the
# wavelength of the perturbation and merge over time.
#
# Setup (Springel 2010 / Athena-style):
#   * Periodic [0, 1]² mesh.
#   * Two-layer flow with smoothed tanh interfaces at y = 1/4 and 3/4
#     so the IC is C^∞ — avoids Gibbs ringing at the discontinuity that
#     would dominate the L2 error before any KH structure develops:
#       ρ(y) = 1 + Δρ · S(y)
#       u_x(y) = u_inner · S(y) + u_outer · (1 − S(y))
#     where S(y) = ½(tanh((y − 0.25)/σ) − tanh((y − 0.75)/σ)).
#   * Initial v-perturbation:
#       u_y(x, y) = δ_v · sin(4π x) · [exp(−(y − 0.25)²/σ²)
#                                     + exp(−(y − 0.75)²/σ²)]
#   * Uniform pressure P_0 = 2.5 (γ = 5/3).
#
# Multi-frame VTU output. Watch vortices form by frame 4-6.
#
# Build / run:
#   mkdir -p /tmp/kh
#   pixi run mojo run -I src examples/euler_kelvin_helmholtz_2d_vtu.mojo

from std.math import exp, tanh, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_2d import dump_euler_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 128
comptime NY: Int = 128
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO_OUTER: Real = 1.0
comptime RHO_INNER: Real = 2.0
comptime U_OUTER: Real = -0.5
comptime U_INNER: Real = +0.5
comptime PRESSURE: Real = 2.5
comptime SIGMA: Real = 0.025          # tanh / Gaussian width
comptime DELTA_V: Real = 0.01         # v-perturbation amplitude
comptime T_FINAL: Real = 2.0
comptime DT: Real = 1.0e-3
comptime NUM_FRAMES: Int = 11         # frames 0 .. 10 inclusive
comptime OUTPUT_DIR: String = "/tmp/kh"
comptime BASENAME: String = "kh"


@always_inline
def _shape(y: Real) -> Real:
    """Smooth indicator for the inner layer: ~1 on (0.25, 0.75), ~0 outside."""
    return Real(0.5) * (tanh((y - Real(0.25)) / SIGMA)
                        - tanh((y - Real(0.75)) / SIGMA))


def _integrate_field[F: Int](
    var ctx: DeviceContext,
    mut q: DGState2D[P, EULER_NC],
    mesh: BlockMesh2D[P],
) raises -> Real:
    var P1 = NodalBasis1D[P].num_nodes
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var s: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * EULER_NC
                        s += ja * weights[in_] * weights[jn] * h[b + F]
    return s


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var S = _shape(y)
                    var rho = RHO_OUTER + (RHO_INNER - RHO_OUTER) * S
                    var ux = U_OUTER + (U_INNER - U_OUTER) * S
                    # v-perturbation localised near the two interfaces.
                    var bump1 = exp(-(y - Real(0.25)) * (y - Real(0.25)) / (SIGMA * SIGMA))
                    var bump2 = exp(-(y - Real(0.75)) * (y - Real(0.75)) / (SIGMA * SIGMA))
                    var uy = DELTA_V * sin(Real(2.0) * TWO_PI * x) * (bump1 + bump2)
                    var ke = Real(0.5) * rho * (ux * ux + uy * uy)
                    var E_tot = PRESSURE / (GAMMA - Real(1.0)) + ke
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * EULER_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = rho * ux
                    host[b + IDX_MY] = rho * uy
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var mass_0 = _integrate_field[IDX_RHO](ctx, q, mesh)
    var e_0 = _integrate_field[IDX_E](ctx, q, mesh)
    print("Kelvin-Helmholtz instability, P=", P, ", N=", NX, "x", NY)
    print("time         ∫ρ dV            ∫E dV            rel Δmass     rel ΔE")
    print("-----------  ---------------  ---------------  ------------  ------------")

    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState2D[P, EULER_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)
    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_euler_2d_frame[P](fw.next_path(t), ctx, mesh, q, GAMMA)
        var mass = _integrate_field[IDX_RHO](ctx, q, mesh)
        var e_tot = _integrate_field[IDX_E](ctx, q, mesh)
        var dm = (mass - mass_0) / mass_0
        var dE = (e_tot - e_0) / e_0
        print(t, " ", mass, " ", e_tot, " ", dm, " ", dE)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step(ctx, q, dt_use, compute_rhs, adjust)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
