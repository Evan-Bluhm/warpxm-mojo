# 2D strong MHD blast — ΔP/P_amb = 1000 spherical pulse in a B₀ = 5
# background — with multi-frame VTU output. Demonstrates the
# `PositivityMHD2D` adjuster in action: without it the simulation
# would NaN almost immediately as the shock fronts drive P_gas
# through zero in the post-shock rarefaction wake.
#
# Per frame the driver prints (min ρ, min P_gas, ∫ρ dV, ∫B_x dV,
# ∫E dV) — mass and ∫B_x are exactly conserved on the periodic
# domain, while min ρ and min P_gas remain positive throughout.
#
# Build / run:
#     mkdir -p /tmp/mhd_strong_blast_2d
#     pixi run mojo run -I src examples/imhd_strong_blast_2d_vtu.mojo
#     # Then open /tmp/mhd_strong_blast_2d/blast.pvd in ParaView.

from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.adjusters.positivity_mhd import PositivityMHD2D
from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_imhd_2d import dump_imhd_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 64
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.05
comptime P_AMB: Real = 0.1
comptime DELTA_P: Real = 100.0
comptime B0: Real = 5.0
comptime T_FINAL: Real = 0.02
comptime DT: Real = 1.0e-4
comptime NUM_FRAMES: Int = 11
comptime OUTPUT_DIR: String = "/tmp/mhd_strong_blast_2d"
comptime BASENAME: String = "blast"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)

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
                    var pp = P_AMB + DELTA_P * exp(-r2 / (SIGMA * SIGMA))
                    var bsq = B0 * B0
                    var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                    var n = jn * P1 + in_
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

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var mass_0: Real = 0.0
    var bx_0: Real = 0.0
    var e_0: Real = 0.0
    for j in range(NY):
        for i in range(NX):
            var el = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var n = jn * P1 + in_
                    var b = (el * nodes_per_elem + n) * MHD_NC
                    var jw = ja * weights[in_] * weights[jn]
                    mass_0 += jw * host[b + IDX_RHO]
                    bx_0 += jw * host[b + IDX_BX]
                    e_0 += jw * host[b + IDX_E]

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_NC](ctx, mesh.num_elements)
    var positivity = PositivityMHD2D[P, MHD_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var fw = FrameWriter(BASENAME, OUTPUT_DIR)

    var frame_dt = T_FINAL / Real(NUM_FRAMES - 1)
    print("frame  time     min ρ        min P_gas    ∫ρ dV       ∫B_x dV     ∫E dV       relΔm       relΔB_x     relΔE")
    print("-----  -------  -----------  -----------  ----------  ----------  ----------  ----------  ----------  ----------")


    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState2D[P, MHD_NC]) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_imhd_2d_frame[P](fw.next_path(t), ctx, mesh, q, GAMMA)

        var min_rho: Real = 1.0e30
        var min_p: Real = 1.0e30
        var mass: Real = 0.0
        var bx_int: Real = 0.0
        var e_int: Real = 0.0
        with q.buf.map_to_host() as h:
            for j in range(NY):
                for i in range(NX):
                    var el = j * NX + i
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = jn * P1 + in_
                            var b = (el * nodes_per_elem + n) * MHD_NC
                            var rho = h[b + IDX_RHO]
                            var mx = h[b + IDX_MX]
                            var my = h[b + IDX_MY]
                            var mz = h[b + IDX_MZ]
                            var E = h[b + IDX_E]
                            var Bx = h[b + IDX_BX]
                            var By = h[b + IDX_BY]
                            var Bz = h[b + IDX_BZ]
                            var ke = Real(0.5) * (mx * mx + my * my + mz * mz) / rho
                            var mag = Real(0.5) * (Bx * Bx + By * By + Bz * Bz)
                            var pp = (GAMMA - Real(1.0)) * (E - ke - mag)
                            if rho < min_rho: min_rho = rho
                            if pp < min_p: min_p = pp
                            var jw = ja * weights[in_] * weights[jn]
                            mass += jw * rho
                            bx_int += jw * Bx
                            e_int += jw * E
        var dm = fabs(mass - mass_0) / mass_0
        var db = fabs(bx_int - bx_0) / bx_0
        var de = fabs(e_int - e_0) / e_0
        print(frame, " ", t, " ", min_rho, " ", min_p, " ", mass, " ", bx_int, " ", e_int, " ", dm, " ", db, " ", de)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step(ctx, q, dt_use, compute_rhs, adjust)
                t += dt_use

    fw.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "frames + PVD collection at", OUTPUT_DIR)
