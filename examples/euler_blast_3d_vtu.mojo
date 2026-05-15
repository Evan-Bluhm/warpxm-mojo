# 3D Euler blast wave with multi-frame VTU output.
#
# A Gaussian pressure pulse at the center of a triply-periodic box drives
# a spherical shock wave outward. Demonstrates the 3D pipeline end-to-end:
#   * BlockMesh3D (periodic 3-axis)
#   * Euler3D with γ-law EOS
#   * DGEulerSolver3D + MoeRossmanith3D[P, EULER_NC] for shock capturing
#   * SSPRK3_Euler_Limited_3D for time integration
#   * FrameWriter + dump_euler_3d_frame for ParaView animation
#
# Per frame the driver prints (∫ρ dV, ∫E dV, max ρ, min ρ) so the user
# can watch conservation hold and the rarefaction at the centre grow.
#
# Build / run:
#     mkdir -p /tmp/euler_blast_3d
#     pixi run mojo run -I src examples/euler_blast_3d_vtu.mojo
#     # Then open /tmp/euler_blast_3d/blast.pvd in ParaView.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith_3d import MoeRossmanith3D
from warpxm.apps.euler_2d import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler_3d import Euler3D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_3d import dump_euler_3d_frame
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime DELTA_P: Real = 5.0
comptime T_FINAL: Real = 0.15
comptime DT: Real = 5.0e-4
comptime NUM_FRAMES: Int = 8
comptime OUTPUT_DIR: String = "/tmp/euler_blast_3d"
comptime BASENAME: String = "blast"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        var zl = mesh.z_min + Real(k) * mesh.dz
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var r2 = (x - Real(0.5)) * (x - Real(0.5))
                            r2 += (y - Real(0.5)) * (y - Real(0.5))
                            r2 += (z - Real(0.5)) * (z - Real(0.5))
                            var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = Real(1.0)
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler3D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, EULER_NC](ctx, mesh.num_elements)
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)

    print("frame    time          ∫ρ dV            ∫E dV            min ρ      max ρ")
    print("-----  ------------  ----------------  ----------------  ---------  ---------")


    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState3D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_euler_3d_frame[P](fw.next_path(t), ctx, mesh, q, GAMMA)
        var mass: Real = 0.0
        var energy: Real = 0.0
        var min_rho: Real = 1.0e30
        var max_rho: Real = 0.0
        with q.buf.map_to_host() as h:
            for k in range(NZ):
                for j in range(NY):
                    for i in range(NX):
                        var e = k * NX * NY + j * NX + i
                        for kn in range(P1):
                            for jn in range(P1):
                                for in_ in range(P1):
                                    var n = kn * P1 * P1 + jn * P1 + in_
                                    var b = (e * nodes_per_elem + n) * EULER_NC
                                    var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                    var rho = h[b + IDX_RHO]
                                    mass += jw * rho
                                    energy += jw * h[b + IDX_E]
                                    if rho < min_rho: min_rho = rho
                                    if rho > max_rho: max_rho = rho
        print(frame, " ", t, " ", mass, " ", energy, " ", min_rho, " ", max_rho)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step[compute_rhs, adjust](ctx, q, dt_use)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
