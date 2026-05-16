# 2D Sedov-like blast with positivity preservation + multi-frame VTU output.
#
# This driver exercises the full strong-shock pipeline:
#   * `SSPRK3_2D` — the generic three-stage Shu-Osher stepper, with a
#     user-supplied `adjust` closure that chains the Moe-Rossmanith
#     limiter and the Zhang-Shu positivity preserver between stages.
#   * VTU output every 0.01 time units so the user can watch the
#     spherically symmetric shock expand in ParaView.
#
# IC: ambient ρ = 1, P_amb = 1e-3 fluid at rest, with a small Gaussian
# pressure pulse at the centre (peak P_inner = 10, so the pressure
# ratio is 10⁴). Without the positivity preserver this NaN's quickly
# as the shock fronts pull P_gas through zero in the rarefaction wake.
#
# The driver prints, per frame, (∫ρ dV, ∫E dV, max |u|, min ρ, min P) —
# mass and total energy are exactly conserved on the periodic domain,
# while min ρ and min P stay positive thanks to the preserver.
#
# Build / run:
#     mkdir -p /tmp/sedov_2d
#     pixi run mojo run -I src examples/euler_sedov_blast_2d_vtu.mojo
#     # Then open /tmp/sedov_2d/sedov.pvd in ParaView.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.adjusters.positivity_euler import PositivityEuler2D
from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_2d import dump_euler_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 64
comptime GAMMA: Real = 1.4
comptime SIGMA: Real = 0.03
comptime P_AMB: Real = 1.0e-3
comptime P_INNER: Real = 10.0           # 10⁴ × ambient
comptime T_FINAL: Real = 0.10
comptime DT: Real = 2.0e-4
comptime NUM_FRAMES: Int = 11           # frames 0..10
comptime OUTPUT_DIR: String = "/tmp/sedov_2d"
comptime BASENAME: String = "sedov"


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
    var weights = NodalBasis1D[P].weights()
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
                    var r2 = (x - Real(0.5)) * (x - Real(0.5))
                    r2 += (y - Real(0.5)) * (y - Real(0.5))
                    var pp = P_AMB + (P_INNER - P_AMB) * exp(-r2 / (SIGMA * SIGMA))
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * EULER_NC
                    host[b + IDX_RHO] = Real(1.0)
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var mass_0: Real = 0.0
    var e_0: Real = 0.0
    for j in range(NY):
        for i in range(NX):
            var el = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var n = jn * P1 + in_
                    var b = (el * nodes_per_elem + n) * EULER_NC
                    var jw = ja * weights[in_] * weights[jn]
                    mass_0 += jw * host[b + IDX_RHO]
                    e_0 += jw * host[b + IDX_E]

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)
    var positivity = PositivityEuler2D[P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState2D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)
    print("frame   time      ∫ρ dV       ∫E dV       max|u|        min ρ          min P_gas      rel Δmass    rel ΔE")
    print("-----  -------  -----------  -----------  -----------  -------------  -------------  -----------  -----------")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_euler_2d_frame[P](fw.next_path(t), ctx, mesh, q, GAMMA)

        var mass: Real = 0.0
        var e_int: Real = 0.0
        var max_u: Real = 0.0
        var min_rho: Real = 1.0e30
        var min_p: Real = 1.0e30
        with q.buf.map_to_host() as h:
            for j in range(NY):
                for i in range(NX):
                    var el = j * NX + i
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = jn * P1 + in_
                            var b = (el * nodes_per_elem + n) * EULER_NC
                            var rho = h[b + IDX_RHO]
                            var mx = h[b + IDX_MX]
                            var my = h[b + IDX_MY]
                            var E = h[b + IDX_E]
                            var u = mx / rho
                            var v = my / rho
                            var ke = Real(0.5) * rho * (u * u + v * v)
                            var pp = (GAMMA - Real(1.0)) * (E - ke)
                            var jw = ja * weights[in_] * weights[jn]
                            mass += jw * rho
                            e_int += jw * E
                            var spd = (u * u + v * v) ** Real(0.5)
                            if spd > max_u: max_u = spd
                            if rho < min_rho: min_rho = rho
                            if pp < min_p: min_p = pp
        var dm = fabs(mass - mass_0) / mass_0
        var de = fabs(e_int - e_0) / e_0
        print(frame, " ", t, " ", mass, " ", e_int, " ", max_u, " ", min_rho, " ", min_p, " ", dm, " ", de)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step[compute_rhs, adjust](ctx, q, dt_use)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
