# 2D Langmuir oscillation — single-fluid plasma with VTU output.
#
# A mobile electron fluid against a static positive-ion background.
# A small density perturbation drives an E_x field that oscillates the
# electrons at the plasma frequency ω_pe = √(α²·n_0/ε_0) = 1.
#
# Setup (same as tests/test_langmuir_2d.mojo but extruded to a thin
# 2D slab so the field can be visualised in ParaView):
#   * 64×8 periodic mesh on [0,1] × [0, 1/8]
#   * α = -1, n_0 = 1, ε_0 = 1
#   * ρ_e(x, 0) = 1 + δ·sin(2π x), δ = 0.01
#   * E_x(x, 0) = (δ/(k·ε_0))·cos(2π x)
#
# Output: two parallel PVD collections per frame — one for the fluid
# state (`langmuir_fluid_NNNN.vtu`) and one for the Maxwell field
# (`langmuir_maxwell_NNNN.vtu`). Both share the same time stamps so
# ParaView can play them in sync.
#
# Build / run:
#   mkdir -p /tmp/langmuir
#   pixi run mojo run -I src examples/langmuir_2d_vtu.mojo
#   # Then open /tmp/langmuir/langmuir_*.pvd in ParaView.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.apps.euler_2d import (
    Euler2D, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.maxwell_2d import (
    Maxwell2D, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_2d import dump_euler_2d_frame
from warpxm.driver.vtu_maxwell_2d import dump_maxwell_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.spatial.lorentz_source_2d import apply_lorentz_source_2d
from warpxm.spatial.maxwell_current_source_2d import apply_current_source_2d
from warpxm.temporal.ssprk3_twofluid_2d import SSPRK3_TwoFluid_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 64
comptime NY: Int = 8
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = -1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime NUM_PERIODS: Real = 2.0           # number of Langmuir periods to integrate
comptime DT: Real = 1.0e-3
comptime NUM_FRAMES: Int = 13              # frames 0 .. 12 — covers two periods
comptime OUTPUT_DIR: String = "/tmp/langmuir"


def _integrate_field[F: Int, NC: Int](
    var ctx: DeviceContext,
    mut q: DGState2D[P, NC],
    mesh: BlockMesh2D[P],
) raises -> Real:
    var P1 = NodalBasis1D[P].num_nodes
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var s: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * NC
                        s += ja * weights[in_] * weights[jn] * h[b + F]
    return s


def _integrate_squared_field[F: Int, NC: Int](
    var ctx: DeviceContext,
    mut q: DGState2D[P, NC],
    mesh: BlockMesh2D[P],
) raises -> Real:
    var P1 = NodalBasis1D[P].num_nodes
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var s: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * NC
                        var v = h[b + F]
                        s += ja * weights[in_] * weights[jn] * v * v
    return s


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_fluid = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1
    var k_wave = TWO_PI

    var fh = ctx.enqueue_create_host_buffer[Real.dtype](q_fluid.buf_len)
    var mh = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var s = sin(k_wave * x)
                    var c_v = cos(k_wave * x)
                    var n = jn * P1 + in_

                    var fb = (e * nodes_per_elem + n) * EULER_NC
                    fh[fb + IDX_RHO] = Real(1.0) + DELTA * s
                    fh[fb + IDX_MX] = Real(0.0)
                    fh[fb + IDX_MY] = Real(0.0)
                    fh[fb + IDX_MZ] = Real(0.0)
                    fh[fb + IDX_E] = P0 / (GAMMA - Real(1.0))

                    var mb = (e * nodes_per_elem + n) * MAXWELL_NC
                    mh[mb + IDX_EX] = (DELTA / (k_wave * EPS0)) * c_v
                    mh[mb + IDX_EY] = Real(0.0)
                    mh[mb + IDX_EZ] = Real(0.0)
                    mh[mb + IDX_BX] = Real(0.0)
                    mh[mb + IDX_BY] = Real(0.0)
                    mh[mb + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q_fluid.buf, src_buf=fh)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=mh)

    var fluid_app = Euler2D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell2D(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver2D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver2D[P](maxwell_app)
    var stepper = SSPRK3_TwoFluid_2D[P](ctx, mesh.num_elements)

    var fw_fluid = FrameWriter("langmuir_fluid", OUTPUT_DIR)
    var fw_maxwell = FrameWriter("langmuir_maxwell", OUTPUT_DIR)

    var omega_p: Real = 1.0       # √(α²·n_0/ε_0) = 1
    var T = TWO_PI / omega_p      # Langmuir period ≈ 6.283
    var t_final = NUM_PERIODS * T

    var mass_0 = _integrate_field[IDX_RHO, EULER_NC](ctx, q_fluid, mesh)
    print("2D Langmuir oscillation, ω_pe =", omega_p, ", T =", T)
    print("time         ∫ρ_e dV          ∫E_x² dV         rel Δmass")
    print("-----------  ---------------  ---------------  ------------")

    @parameter
    def compute_joint_rhs(
        mut qf_in: DGState2D[P, EULER_NC],
        mut qm_in: DGState2D[P, MAXWELL_NC],
        mut qf_out: DGState2D[P, EULER_NC],
        mut qm_out: DGState2D[P, MAXWELL_NC],
    ) raises:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source_2d[P](ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source_2d[P](ctx, qf_in, qm_out, ALPHA, EPS0)

    var frame_dt = t_final / Real(NUM_FRAMES - 1)
    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        ctx.synchronize()
        dump_euler_2d_frame[P](fw_fluid.next_path(t), ctx, mesh, q_fluid, GAMMA)
        dump_maxwell_2d_frame[P](fw_maxwell.next_path(t), ctx, mesh, q_maxwell)
        var mass = _integrate_field[IDX_RHO, EULER_NC](ctx, q_fluid, mesh)
        var ex2 = _integrate_squared_field[IDX_EX, MAXWELL_NC](ctx, q_maxwell, mesh)
        var dm = (mass - mass_0) / mass_0
        print(t, " ", mass, " ", ex2, " ", dm)

        if frame < NUM_FRAMES - 1:
            var t_target = Real(frame + 1) * frame_dt
            while t < t_target - Real(0.5) * DT:
                var dt_use = DT if (t + DT) < t_target else (t_target - t)
                stepper.step[compute_joint_rhs](ctx, q_fluid, q_maxwell, dt_use)
                t += dt_use

    fw_fluid.write_pvd()
    fw_maxwell.write_pvd()
    print("")
    print("Wrote", NUM_FRAMES, "fluid +", NUM_FRAMES, "Maxwell frames at", OUTPUT_DIR)
    print("Open langmuir_fluid.pvd or langmuir_maxwell.pvd in ParaView.")
