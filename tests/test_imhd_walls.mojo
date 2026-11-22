# Ideal-MHD wall boundary conditions in 2D and 3D.
#
# A closed box `[0, 1]^d` is filled with a uniform state whose velocity and
# magnetic field both have a non-zero component normal to every wall:
#
#     ρ = 1,  p = 1,  u = (0.10, 0.07, 0.05),  B = (0.10, 0.07, 0.03)
#
# The flow piles up against the walls, so the ghost state synthesised by
# `_apply_bc_imhd{,_3d}` matters on every boundary face from the first step.
# The checks are exact discrete identities of the Rusanov flux with a
# mirrored ghost state, not accuracy estimates:
#
#   * `BC_WALL_FREESLIP` (normal momentum flipped): the mass flux through
#     the wall is `½(ρu_n − ρu_n) = 0`, so total mass is conserved to
#     round-off. A copy-out ghost would leak `ρ u_n` per unit area.
#   * `BC_PEC_WALL` (normal momentum and normal B flipped): mass flux is
#     zero as above, and the energy flux `(E + p*)u_n − B_n (u·B)` is odd
#     under the mirror, so total energy is conserved to round-off too.
#     With only the momentum flipped the wall would pass an energy flux
#     `−B_n (u_t·B_t)`; the free-slip run checks that this is what happens,
#     i.e. that the two BC kinds are really distinct.
#   * In both cases every wall pair must actually push back: the total
#     momentum along each walled axis changes by an O(1) fraction of its
#     initial value. (With a copy-out ghost the uniform state is an exact
#     steady solution and nothing changes at all.)
#
# Build / run:
#     pixi run mojo run -I src tests/test_imhd_walls.mojo

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime N2: Int = 16          # elements per side, 2D
comptime N3: Int = 8           # elements per side, 3D
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO0: Real = 1.0
comptime P0: Real = 1.0
comptime UX: Real = 0.10
comptime UY: Real = 0.07
comptime UZ: Real = 0.05
comptime BX: Real = 0.10
comptime BY: Real = 0.07
comptime BZ: Real = 0.03
comptime N_STEPS: Int = 100
comptime DT: Real = 2.0e-3     # t_final = 0.2


def _uniform() -> Array[Real, MHD_NC]:
    var v = Array[Real, MHD_NC](fill=Real(0.0))
    v[IDX_RHO] = RHO0
    v[IDX_MX] = RHO0 * UX
    v[IDX_MY] = RHO0 * UY
    v[IDX_MZ] = RHO0 * UZ
    v[IDX_E] = (
        P0 / (GAMMA - Real(1.0))
        + Real(0.5) * RHO0 * (UX * UX + UY * UY + UZ * UZ)
        + Real(0.5) * (BX * BX + BY * BY + BZ * BZ)
    )
    v[IDX_BX] = BX
    v[IDX_BY] = BY
    v[IDX_BZ] = BZ
    return v^


def _check(
    label: String, bc: Int,
    mass0: Real, mass1: Real, en0: Real, en1: Real,
    mx0: Real, mx1: Real, my0: Real, my1: Real, mz0: Real, mz1: Real,
    has_z_walls: Bool,
) raises:
    var d_mass = mass1 - mass0
    if d_mass < Real(0.0): d_mass = -d_mass
    var d_en = en1 - en0
    if d_en < Real(0.0): d_en = -d_en
    var d_mx = mx1 - mx0
    if d_mx < Real(0.0): d_mx = -d_mx
    print(
        label, "PEC wall:" if bc == BC_PEC_WALL else "free-slip wall:",
        " |Δmass| =", d_mass, " |ΔE| =", d_en, " |Δ∫ρu_x| =", d_mx,
    )
    assert_true(mass1 == mass1 and en1 == en1, "wall run produced NaN")
    assert_true(d_mass < Real(1.0e-12), "mass leaked through the wall")
    assert_true(
        d_mx > Real(0.2) * mx0,
        "x-momentum barely changed — the x walls are not acting",
    )
    var d_my = my1 - my0
    if d_my < Real(0.0): d_my = -d_my
    assert_true(
        d_my > Real(0.2) * my0,
        "y-momentum barely changed — the y walls are not acting",
    )
    if has_z_walls:
        var d_mz = mz1 - mz0
        if d_mz < Real(0.0): d_mz = -d_mz
        assert_true(
            d_mz > Real(0.2) * mz0,
            "z-momentum barely changed — the z walls are not acting",
        )
    if bc == BC_PEC_WALL:
        assert_true(d_en < Real(1.0e-12), "energy leaked through the PEC wall")
    else:
        assert_true(
            d_en > Real(1.0e-5),
            "free-slip wall conserved energy exactly — it is flipping B_n too",
        )


def _run_2d(var ctx: DeviceContext, bc: Int) raises:
    comptime State = DGState[2, P, MHD_NC]
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=N2, ny=N2, periodic_x=False, periodic_y=False,
        bc_left=bc, bc_right=bc, bc_bottom=bc, bc_top=bc,
    )
    var q = State(ctx, mesh.num_elements)

    def uniform_ic(x: Real, y: Real) -> Array[Real, MHD_NC]:
        return _uniform()
    mesh.fill(q, uniform_ic)

    var spatial = DGImhdSolver[2, P](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3[2, P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, no_op_adjust[2, P, MHD_NC])
    ctx.synchronize()

    # Unit box, uniform IC: the initial integrals are the nodal values.
    var uniform = _uniform()
    _check(
        "2D", bc,
        uniform[IDX_RHO], mesh.integrate(q, IDX_RHO),
        uniform[IDX_E], mesh.integrate(q, IDX_E),
        uniform[IDX_MX], mesh.integrate(q, IDX_MX),
        uniform[IDX_MY], mesh.integrate(q, IDX_MY),
        uniform[IDX_MZ], mesh.integrate(q, IDX_MZ),
        False,
    )


def _run_3d(var ctx: DeviceContext, bc: Int) raises:
    comptime State = DGState[3, P, MHD_NC]
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=N3, ny=N3, nz=N3,
        periodic_x=False, periodic_y=False, periodic_z=False,
        bc_left=bc, bc_right=bc, bc_bottom=bc, bc_top=bc,
        bc_down=bc, bc_up=bc,
    )
    var q = State(ctx, mesh.num_elements)

    def uniform_ic(x: Real, y: Real, z: Real) -> Array[Real, MHD_NC]:
        return _uniform()
    mesh.fill(q, uniform_ic)

    var spatial = DGImhdSolver[3, P](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3[3, P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, no_op_adjust[3, P, MHD_NC])
    ctx.synchronize()

    # Unit box, uniform IC: the initial integrals are the nodal values.
    var uniform = _uniform()
    _check(
        "3D", bc,
        uniform[IDX_RHO], mesh.integrate(q, IDX_RHO),
        uniform[IDX_E], mesh.integrate(q, IDX_E),
        uniform[IDX_MX], mesh.integrate(q, IDX_MX),
        uniform[IDX_MY], mesh.integrate(q, IDX_MY),
        uniform[IDX_MZ], mesh.integrate(q, IDX_MZ),
        True,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _run_2d(ctx, BC_WALL_FREESLIP)
    _run_2d(ctx, BC_PEC_WALL)
    _run_3d(ctx, BC_WALL_FREESLIP)
    _run_3d(ctx, BC_PEC_WALL)
    print("PASSED")
