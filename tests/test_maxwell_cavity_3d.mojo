# 3D rectangular cavity TM mode regression.
#
# Same TM₁₁ standing-wave family as the 2D test, but on a 3D mesh bounded
# by PEC walls on all six sides. We pick a z-uniform mode (the simplest
# analytically valid eigenmode of a [0,1]³ PEC cavity that doesn't require
# more elaborate construction):
#
#     E_z(x, y, z, t) = sin(π x) · sin(π y) · cos(ω t)
#     B_x = +(π / ω) · sin(π x) · cos(π y) · sin(ω t)
#     B_y = -(π / ω) · cos(π x) · sin(π y) · sin(ω t)
#     (E_x = E_y = B_z = 0 throughout)
#
# with `ω = c · π · √2`. After one full period `T = 2π/ω = √2 / c` the
# fields return to the IC.
#
# PEC boundary conditions are satisfied by construction:
#   * x = 0 or 1: tangential E (here E_z, E_y) vanishes via the sin(πx)
#     factor; normal B (B_x) vanishes via the same sin(πx) factor
#   * y = 0 or 1: analogous via sin(πy)
#   * z = 0 or 1: tangential E (E_x, E_y) is zero throughout the cavity,
#     normal B (B_z) is zero throughout — all trivially compatible
#
# This setup exercises every PEC wall axis (the x and y walls actively,
# the z walls trivially) and confirms the 3D Maxwell kernel doesn't leak
# energy into E_x / E_y / B_z modes through the PEC ghost-state synthesis.
#
# A second run covers `BC_PMC_WALL` with the dual problem (E -> cB,
# cB -> -E): the same profile on B_z between PMC walls, where E_z, B_x and
# B_y are the components that must stay zero.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 4
comptime C_LIGHT: Real = 1.0
comptime State = DGState[3, P, MAXWELL_NC]


def _run_cavity(var ctx: DeviceContext, pmc: Bool) raises:
    """One period of the cavity mode. `pmc=False`: PEC walls, E_z mode.
    `pmc=True`: the dual problem (E -> cB, cB -> -E) — PMC walls with the
    same profile on B_z, so E_z, B_x, B_y are the unexcited components."""
    var bc = BC_PMC_WALL if pmc else BC_PEC_WALL
    var i_mode = IDX_BZ if pmc else IDX_EZ
    var i_zero0 = IDX_BX if pmc else IDX_EX
    var i_zero1 = IDX_BY if pmc else IDX_EY
    var i_zero2 = IDX_EZ if pmc else IDX_BZ
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=False, periodic_z=False,
        bc_left=bc, bc_right=bc,
        bc_bottom=bc, bc_top=bc,
        bc_down=bc, bc_up=bc,
    )
    var q = State(ctx, mesh.num_elements)

    # IC: only the mode component, at the peak of the cos(ωt) factor.
    def initial(x: Real, y: Real, z: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[i_mode] = sin(PI * x) * sin(PI * y)
        return v^
    mesh.fill(q, initial)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver[3, P](app)
    var stepper = SSPRK3[3, P, MAXWELL_NC](ctx, mesh.num_elements)

    var t_final = sqrt(Real(2.0)) / C_LIGHT
    # dt divides the period exactly (≈ 5e-4, the step the plane-wave tests
    # use on the same mesh), so the run ends exactly at t = T.
    var dt: Real = t_final / Real(2829)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs,
        no_op_adjust[3, P, MAXWELL_NC],
    )
    ctx.synchronize()

    def mode(x: Real, y: Real, z: Real) -> Real:
        return sin(PI * x) * sin(PI * y)
    var l2_ez = mesh.l2_error(q, i_mode, mode)
    # The three components outside the mode must remain identically zero.
    var max_other: Real = 0.0
    def scan(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, MAXWELL_NC],
    ) {mut max_other, imm}:
        for idx in [i_zero0, i_zero1, i_zero2]:
            max_other = fmax(max_other, fabs(v[idx]))
    mesh.visit_nodes(q, scan)
    print("3D Maxwell cavity,", "PMC walls, B_z mode" if pmc else "PEC walls, E_z mode", "— t =", t)
    print("  L2 mode error vs IC:                    ", l2_ez)
    print("  max |unexcited components| (must be ~0):", max_other)
    assert_true(l2_ez < Real(2.0e-4), "mode L2 error too large — wall reflection wrong")
    assert_true(max_other < Real(1.0e-10),
                "unexcited components grew — wall BC mixing modes")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _run_cavity(ctx, False)
    _run_cavity(ctx, True)
    print("PASSED")
