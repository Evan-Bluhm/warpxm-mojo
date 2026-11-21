# 3D MHD blast wave with multi-frame VTU output.
#
# Spherical pressure pulse in a uniform magnetic background — the fluid
# drags the frozen-in field as the shock expands. Because the IC has
# uniform B and zero current, ∇·B = 0 is preserved exactly under ideal
# MHD without needing GLM cleaning.
#
# Per frame the driver prints (∫ρ dV, ∫B_x dV, ∫E dV) and their drifts —
# mass and ∫B_x are exactly conserved (periodic + Faraday's law), so any
# drift indicates numerical leakage.
#
# Build / run:
#     mkdir -p /tmp/imhd_blast_3d
#     pixi run mojo run -I src examples/imhd_blast_3d_vtu.mojo
#     # Then open /tmp/imhd_blast_3d/blast.pvd in ParaView.

from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.imhd import IMHD
from warpxm.apps.imhd import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_imhd_3d import dump_imhd_3d_frame
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime DELTA_P: Real = 5.0
comptime B0: Real = 1.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 5.0e-4
comptime NUM_FRAMES: Int = 11
comptime OUTPUT_DIR: String = "/tmp/imhd_blast_3d"
comptime BASENAME: String = "blast"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)

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
                            var bsq = B0 * B0
                            var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_NC
                            host[b + IDX_RHO] = Real(1.0)
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = E_tot
                            host[b + IDX_BX] = B0
                            host[b + IDX_BY] = Real(0.0)
                            host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver3D[P](app)
    var stepper = SSPRK3_3D[P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, MHD_NC](ctx, mesh.num_elements)
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)

    var mass_0: Real = 0.0
    var bx_0: Real = 0.0
    var e_0: Real = 0.0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_NC
                            var jw = jv * weights[in_] * weights[jn] * weights[kn]
                            mass_0 += jw * host[b + IDX_RHO]
                            bx_0 += jw * host[b + IDX_BX]
                            e_0 += jw * host[b + IDX_E]

    print("frame   time      ∫ρ dV         ∫B_x dV       ∫E dV         rel Δmass    rel ΔB_x      rel ΔE")
    print("-----  -------  -------------  -------------  -------------  -----------  -----------  -----------")


    def compute_rhs(
        mut q_in: DGState3D[P, MHD_NC],
        mut q_out: DGState3D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState3D[P, MHD_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_imhd_3d_frame[P](fw.next_path(t), ctx, mesh, q, GAMMA)

        var mass: Real = 0.0
        var bx_int: Real = 0.0
        var e_int: Real = 0.0
        with q.buf.map_to_host() as h:
            for k in range(NZ):
                for j in range(NY):
                    for i in range(NX):
                        var e = k * NX * NY + j * NX + i
                        for kn in range(P1):
                            for jn in range(P1):
                                for in_ in range(P1):
                                    var n = kn * P1 * P1 + jn * P1 + in_
                                    var b = (e * nodes_per_elem + n) * MHD_NC
                                    var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                    mass += jw * h[b + IDX_RHO]
                                    bx_int += jw * h[b + IDX_BX]
                                    e_int += jw * h[b + IDX_E]
        var dm = (mass - mass_0) / mass_0
        var db = (bx_int - bx_0) / bx_0
        var de = (e_int - e_0) / e_0
        print(frame, " ", t, " ", mass, " ", bx_int, " ", e_int, " ", dm, " ", db, " ", de)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step(ctx, q, dt_use, compute_rhs, adjust)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
