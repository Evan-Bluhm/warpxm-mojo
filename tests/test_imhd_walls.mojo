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

from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_WALL_FREESLIP, BC_PEC_WALL
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_2d import DGState2D
from warpxm.variable.state_3d import DGState3D


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


def _fill_uniform(mut host: Array[Real, MHD_NC]):
    host[IDX_RHO] = RHO0
    host[IDX_MX] = RHO0 * UX
    host[IDX_MY] = RHO0 * UY
    host[IDX_MZ] = RHO0 * UZ
    host[IDX_E] = (
        P0 / (GAMMA - Real(1.0))
        + Real(0.5) * RHO0 * (UX * UX + UY * UY + UZ * UZ)
        + Real(0.5) * (BX * BX + BY * BY + BZ * BZ)
    )
    host[IDX_BX] = BX
    host[IDX_BY] = BY
    host[IDX_BZ] = BZ


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
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=N2, ny=N2, periodic_x=False, periodic_y=False,
        bc_left=bc, bc_right=bc, bc_bottom=bc, bc_top=bc,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)
    var P1 = NodalBasis1D[P].num_nodes
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var uniform = Array[Real, MHD_NC](fill=Real(0.0))
    _fill_uniform(uniform)
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for n in range(mesh.num_elements * nodes_per_elem):
        for c in range(MHD_NC):
            host[n * MHD_NC + c] = uniform[c]
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGImhdSolver2D[P](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState2D[P, MHD_NC]) raises {imm}:
        pass

    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, no_adjust)
    ctx.synchronize()

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var mass: Real = 0.0
    var en: Real = 0.0
    var mx: Real = 0.0
    var my: Real = 0.0
    var mz: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            for jn in range(P1):
                for in_ in range(P1):
                    var b = (e * nodes_per_elem + jn * P1 + in_) * MHD_NC
                    var jw = ja * weights[in_] * weights[jn]
                    mass += jw * h[b + IDX_RHO]
                    en += jw * h[b + IDX_E]
                    mx += jw * h[b + IDX_MX]
                    my += jw * h[b + IDX_MY]
                    mz += jw * h[b + IDX_MZ]
    # Unit box, uniform IC: the initial integrals are the nodal values.
    _check(
        "2D", bc, uniform[IDX_RHO], mass, uniform[IDX_E], en,
        uniform[IDX_MX], mx, uniform[IDX_MY], my, uniform[IDX_MZ], mz, False,
    )


def _run_3d(var ctx: DeviceContext, bc: Int) raises:
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=N3, ny=N3, nz=N3,
        periodic_x=False, periodic_y=False, periodic_z=False,
        bc_left=bc, bc_right=bc, bc_bottom=bc, bc_top=bc,
        bc_down=bc, bc_up=bc,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)
    var P1 = NodalBasis1D[P].num_nodes
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var uniform = Array[Real, MHD_NC](fill=Real(0.0))
    _fill_uniform(uniform)
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for n in range(mesh.num_elements * nodes_per_elem):
        for c in range(MHD_NC):
            host[n * MHD_NC + c] = uniform[c]
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGImhdSolver3D[P](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = SSPRK3_3D[P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState3D[P, MHD_NC],
        mut q_out: DGState3D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState3D[P, MHD_NC]) raises {imm}:
        pass

    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, no_adjust)
    ctx.synchronize()

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var mass: Real = 0.0
    var en: Real = 0.0
    var mx: Real = 0.0
    var my: Real = 0.0
    var mz: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            for kn in range(P1):
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = kn * P1 * P1 + jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_NC
                        var jw = jv * weights[in_] * weights[jn] * weights[kn]
                        mass += jw * h[b + IDX_RHO]
                        en += jw * h[b + IDX_E]
                        mx += jw * h[b + IDX_MX]
                        my += jw * h[b + IDX_MY]
                        mz += jw * h[b + IDX_MZ]
    _check(
        "3D", bc, uniform[IDX_RHO], mass, uniform[IDX_E], en,
        uniform[IDX_MX], mx, uniform[IDX_MY], my, uniform[IDX_MZ], mz, True,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _run_2d(ctx, BC_WALL_FREESLIP)
    _run_2d(ctx, BC_PEC_WALL)
    _run_3d(ctx, BC_WALL_FREESLIP)
    _run_3d(ctx, BC_PEC_WALL)
    print("PASSED")
