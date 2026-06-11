# Steady-state stepping driver for nsys / ncu profiling of the 3D MHD
# hot path. Uniform IC (cheap to construct) — what matters is the per-step
# DG cost, not the physics setup.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.imhd import IMHD
from warpxm.apps.imhd import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime B0: Real = 1.0
comptime DT: Real = 5.0e-4
comptime WARMUP: Int = 5
comptime N_STEPS: Int = 30


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ, periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        var zl = mesh.z_min + Real(k) * mesh.dz
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = (k * NY + j) * NX + i
                for kn in range(P1):
                    var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var r2 = (x - Real(0.5)) ** 2 + (y - Real(0.5)) ** 2 + (z - Real(0.5)) ** 2
                            var P_loc = P_AMB + Real(5.0) * exp(-r2 / (SIGMA * SIGMA))
                            var E_tot = P_loc / (GAMMA - Real(1.0)) + Real(0.5) * B0 * B0
                            var n = (kn * P1 + jn) * P1 + in_
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

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, MHD_NC],
        mut q_out: DGState3D[P, MHD_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState3D[P, MHD_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    for _ in range(WARMUP):
        stepper.step[compute_rhs, adjust](ctx, q, DT)
    ctx.synchronize()
    print("warmup_done")

    for _ in range(N_STEPS):
        stepper.step[compute_rhs, adjust](ctx, q, DT)
    ctx.synchronize()
    print("measure_done", N_STEPS, "steps NX=", NX, "NY=", NY, "NZ=", NZ, "P=", P)
