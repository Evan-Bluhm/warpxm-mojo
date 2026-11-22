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

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 32
comptime C_LIGHT: Real = 1.0
comptime L: Real = 1.0
comptime DT: Real = 5.0e-4
comptime State = DGState[1, P, MAXWELL_NC]


def _run_cavity(var ctx: DeviceContext, pmc: Bool) raises:
    """One period of the half-wave cavity mode. `pmc=False`: PEC walls,
    E_z = sin(kx). `pmc=True`: the dual problem — PMC walls (tangential B
    vanishes) with B_y = sin(kx), and E_z as the partner field."""
    var bc = BC_PMC_WALL if pmc else BC_PEC_WALL
    var i_mode = IDX_BY if pmc else IDX_EZ
    var i_partner = IDX_EZ if pmc else IDX_BY
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), L, NX,
        periodic=False,
        bc_left=bc, bc_right=bc,
    )
    var q = State(ctx, mesh.num_elements)

    var k_wave: Real = PI / L
    var omega = C_LIGHT * k_wave
    var t_final = Real(2.0) * PI / omega       # one full period

    def initial(x: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[i_mode] = sin(k_wave * x)
        return v^
    mesh.fill(q, initial)

    var spatial = DGMaxwellSolver[1, P](Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)))
    var stepper = SSPRK3[1, P, MAXWELL_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t = stepper.run_until(
        ctx, q, Real(0.0), t_final, DT, compute_rhs,
        no_op_adjust[1, P, MAXWELL_NC],
    )
    ctx.synchronize()

    # After one period: E_z back to IC, B_y back to zero.
    # E_x, E_y, B_x, B_z must remain ~0 throughout (symmetry).
    def mode(x: Real) {imm} -> Real:
        return sin(k_wave * x)
    var l2_Ez = mesh.l2_error(q, i_mode, mode)
    var max_By: Real = 0.0
    var max_other: Real = 0.0
    def scan(x: Real, w: Real, v: Array[Real, MAXWELL_NC]) {mut max_By, mut max_other, imm}:
        max_By = fmax(max_By, fabs(v[i_partner]))
        for idx in range(MAXWELL_NC):
            if idx != i_mode and idx != i_partner:
                max_other = fmax(max_other, fabs(v[idx]))
    mesh.visit_nodes(q, scan)
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
