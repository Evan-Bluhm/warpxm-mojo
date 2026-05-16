# 2D rectangular cavity TM₁₁ mode with multi-frame VTU output.
#
# Standing-wave EM mode in a [0,1]² PEC-walled cavity:
#     E_z(x, y, t) = sin(π x) · sin(π y) · cos(ω t)
#     B_x ∝ sin(π x) · cos(π y) · sin(ω t)
#     B_y ∝ cos(π x) · sin(π y) · sin(ω t)
# with ω = c · π · √2, period T = √2 / c. After one full period the
# fields return to the IC.
#
# Per frame the driver prints (∫E_energy dV, ∫B_energy dV) — these should
# oscillate out of phase as energy sloshes between the electric and
# magnetic field, with the sum stable.
#
# Build / run:
#     mkdir -p /tmp/maxwell_cavity
#     pixi run mojo run -I src examples/maxwell_cavity_2d_vtu.mojo
#     # Then open /tmp/maxwell_cavity/cavity.pvd in ParaView.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_PEC_WALL
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, PI
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_maxwell_2d import dump_maxwell_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime C_LIGHT: Real = 1.0
comptime NUM_FRAMES: Int = 13       # one period split into 12 intervals
comptime OUTPUT_DIR: String = "/tmp/maxwell_cavity"
comptime BASENAME: String = "cavity"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
        bc_bottom=BC_PEC_WALL, bc_top=BC_PEC_WALL,
    )
    var q = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

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
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MAXWELL_NC
                    host[b + IDX_EX] = Real(0.0)
                    host[b + IDX_EY] = Real(0.0)
                    host[b + IDX_EZ] = sin(PI * x) * sin(PI * y)
                    host[b + IDX_BX] = Real(0.0)
                    host[b + IDX_BY] = Real(0.0)
                    host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MAXWELL_NC](ctx, mesh.num_elements)
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var t_final = sqrt(Real(2.0)) / C_LIGHT
    var frame_dt = t_final / Real(NUM_FRAMES - 1)
    var dt: Real = Real(2.0e-4)

    var ja = mesh.jacobian_x * mesh.jacobian_y

    print("frame    time        ∫E_energy dV     ∫B_energy dV     ∫total dV")
    print("-----  ----------  ----------------  ----------------  ----------------")


    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MAXWELL_NC],
        mut q_out: DGState2D[P, MAXWELL_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState2D[P, MAXWELL_NC]) raises:
        pass

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_maxwell_2d_frame[P](fw.next_path(t), ctx, mesh, q)

        var e_energy: Real = 0.0
        var b_energy: Real = 0.0
        with q.buf.map_to_host() as h:
            for j in range(NY):
                for i in range(NX):
                    var e = j * NX + i
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MAXWELL_NC
                            var jw = ja * weights[in_] * weights[jn]
                            var Ex = h[b + IDX_EX]
                            var Ey = h[b + IDX_EY]
                            var Ez = h[b + IDX_EZ]
                            var Bx = h[b + IDX_BX]
                            var By = h[b + IDX_BY]
                            var Bz = h[b + IDX_BZ]
                            e_energy += jw * Real(0.5) * (Ex * Ex + Ey * Ey + Ez * Ez)
                            b_energy += jw * Real(0.5) * (Bx * Bx + By * By + Bz * Bz)
        print(frame, " ", t, " ", e_energy, " ", b_energy, " ", e_energy + b_energy)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * dt:
                var dt_use = dt if (t + dt) < t_target else (t_target - t)
                stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
