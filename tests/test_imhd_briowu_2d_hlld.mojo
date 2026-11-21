# Brio-Wu MHD shock tube with the HLLD numerical flux (Miyoshi-Kusano).
# Mirrors `test_imhd_briowu_2d.mojo` exactly except the spatial solver is
# instantiated with `USE_HLLD=True`.  Validates that the HLLD path
# survives the full t = 0.1 run with the same correctness criteria.

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 4
comptime GAMMA: Real = 2.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 1.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var rho: Real
                    var pp: Real
                    var By: Real
                    if x < Real(0.5):
                        rho = 1.0; pp = 1.0; By = 1.0
                    else:
                        rho = 0.125; pp = 0.1; By = -1.0
                    var Bx_v: Real = 0.75
                    var bsq = Bx_v * Bx_v + By * By
                    var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
                    host[b + IDX_BX] = Bx_v
                    host[b + IDX_BY] = By
                    host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
    var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_NC](ctx, mesh.num_elements)


    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState2D[P, MHD_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var max_y_drift: Real = 0.0
    var rho_row0 = Array[Real, NX](fill=Real(0.0))
    var max_grad: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                var b = (e * nodes_per_elem + mid_node) * MHD_NC
                if isnan(h[b + IDX_RHO]) or isnan(h[b + IDX_E]):
                    any_nan = True
                if j == 0:
                    rho_row0[i] = h[b + IDX_RHO]
                else:
                    var d = fabs(h[b + IDX_RHO] - rho_row0[i])
                    if d > max_y_drift: max_y_drift = d
        # Sharpness proxy: max gradient of ρ along x. HLLD should give a
        # larger value (steeper transitions) than Rusanov.
        for i in range(NX - 1):
            var g = fabs(rho_row0[i + 1] - rho_row0[i])
            if g > max_grad: max_grad = g

    assert_true(not any_nan, "HLLD Brio-Wu produced NaN")
    print("HLLD Brio-Wu — t =", t)
    print("  max y-drift in ρ:   ", max_y_drift)
    print("  max |Δρ/Δx| proxy:  ", max_grad)
    assert_true(max_y_drift < Real(1.0e-10),
                "result should be uniform in y for a 1D-extruded Brio-Wu")
    var ix01 = Int(0.05 / mesh.dx)
    var ix099 = Int(0.95 / mesh.dx)
    print("  ρ at x≈0.05:        ", rho_row0[ix01])
    print("  ρ at x≈0.95:        ", rho_row0[ix099])
    assert_true(fabs(rho_row0[ix01] - Real(1.0)) < Real(2.0e-2),
                "left far-field density off")
    assert_true(fabs(rho_row0[ix099] - Real(0.125)) < Real(2.0e-2),
                "right far-field density off")
    print("PASSED")
