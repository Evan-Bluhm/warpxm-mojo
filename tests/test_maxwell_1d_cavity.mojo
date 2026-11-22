# 1D Maxwell PEC cavity test — half-wavelength standing wave between
# two perfectly-conducting walls.
#
# IC: TM mode with E along ẑ, half-wave in [0, L]:
#     E_z(x, 0) = sin(π x / L)
#     B_y(x, 0) = 0
# Exact standing wave: E_z(x, t) = sin(kx)·cos(ωt),  B_y = -(1/c)·cos(kx)·sin(ωt)
# with k = π/L, ω = c·k = π·c/L.
#
# At PEC walls (x = 0 and x = L), tangential E must vanish — the IC
# already satisfies this. The wall BC handler flips tangential E and
# normal B in the ghost cell, which is what PEC requires.
#
# Period T = 2π/ω = 2L/c. After one period the field returns to IC.
# Validates:
#   - 1D Maxwell BC handler for BC_PEC_WALL.
#   - Walls reflect cleanly — no leakage of energy or spurious mode mixing.
#   - BC_PMC_WALL via the dual problem (B_y = sin(kx) between PMC walls).

from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_PEC_WALL, BC_PMC_WALL
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NX: Int = 32
comptime C_LIGHT: Real = 1.0
comptime L: Real = 1.0
comptime PI: Real = 3.141592653589793
comptime DT: Real = 5.0e-4


def _run_cavity(var ctx: DeviceContext, pmc: Bool) raises:
    """One period of the half-wave cavity mode. `pmc=False`: PEC walls,
    E_z = sin(kx). `pmc=True`: the dual problem — PMC walls (tangential B
    vanishes) with B_y = sin(kx), and E_z as the partner field."""
    var bc = BC_PMC_WALL if pmc else BC_PEC_WALL
    var i_mode = IDX_BY if pmc else IDX_EZ
    var i_partner = IDX_EZ if pmc else IDX_BY
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), L, NX,
        periodic=False,
        bc_left=bc, bc_right=bc,
    )
    var q = DGState1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var k_wave: Real = PI / L
    var omega = C_LIGHT * k_wave
    var t_final = Real(2.0) * PI / omega       # one full period

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var s = sin(k_wave * x)
            var b = (e * nodes_per_elem + i) * MAXWELL_NC
            for c in range(MAXWELL_NC):
                host[b + c] = Real(0.0)
            host[b + i_mode] = s
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGMaxwellSolver1D[P](Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)))
    var stepper = SSPRK3_1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState1D[P, MAXWELL_NC],
        mut q_out: DGState1D[P, MAXWELL_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState1D[P, MAXWELL_NC]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    # After one period: E_z back to IC, B_y back to zero.
    # E_x, E_y, B_x, B_z must remain ~0 throughout (symmetry).
    var err_Ez: Real = 0.0
    var max_By: Real = 0.0
    var max_other: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var s = sin(k_wave * x)
                var b = (e * nodes_per_elem + i) * MAXWELL_NC
                var d_Ez = h[b + i_mode] - s
                err_Ez += mesh.jacobian * weights[i] * d_Ez * d_Ez
                if fabs(h[b + i_partner]) > max_By: max_By = fabs(h[b + i_partner])
                for idx in range(MAXWELL_NC):
                    if idx != i_mode and idx != i_partner:
                        var v = h[b + idx]
                        if fabs(v) > max_other: max_other = fabs(v)
    var l2_Ez = err_Ez ** Real(0.5)
    print(
        "1D Maxwell cavity,", "PMC walls, B_y mode" if pmc else "PEC walls, E_z mode",
        " t =", t, " (period 2L/c =", t_final, "):",
    )
    print("  L2 mode error vs IC             :", l2_Ez)
    print("  max |partner field| (should be ~0):", max_By)
    print("  max |four unexcited components|   :", max_other)
    assert_true(l2_Ez < Real(2.5e-5), "standing-wave error too large")
    assert_true(max_By < Real(6.0e-5), "partner field did not return to ~0 after one period")
    assert_true(max_other < Real(1.0e-12),
                "Unexcited components grew — 1D Maxwell BC handler buggy")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _run_cavity(ctx, False)
    _run_cavity(ctx, True)
    print("PASSED")
