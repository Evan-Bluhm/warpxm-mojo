# 2D rectangular cavity TM₁₁ mode with multi-frame VTU output.
#
# Standing-wave EM mode in a [0,1]² PEC-walled cavity:
#     E_z(x, y, t) = sin(π x) · sin(π y) · cos(ω t)
#     B_x ∝ sin(π x) · cos(π y) · sin(ω t)
#     B_y ∝ cos(π x) · sin(π y) · sin(ω t)
# with ω = c · π · √2, period T = √2 / c. After one full period the
# fields return to the IC.
#
# Per frame the driver prints (∫E_energy dV, ∫B_energy dV) — these should
# oscillate out of phase as energy sloshes between the electric and
# magnetic field, with the sum stable.
#
# Light speed is constant, so the CFL time step is fixed for the run.
#
#     pixi run example maxwell_cavity_2d_vtu
#     # then open output/maxwell_cavity/cavity.pvd in ParaView

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime C_LIGHT: Real = 1.0
comptime NUM_FRAMES: Int = 13       # one period split into 12 intervals
comptime State = DGState[2, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
        bc_bottom=BC_PEC_WALL, bc_top=BC_PEC_WALL,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EZ] = sin(PI * x) * sin(PI * y)
        return v^
    mesh.fill(q, initial)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver[2, P](app)
    var stepper = SSPRK3[2, P, MAXWELL_NC](ctx, mesh.num_elements, mesh.num_ghost)
    var fw = FrameWriter("cavity", "output/maxwell_cavity")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def e_energy(v: Array[Real, MAXWELL_NC]) -> Real:
        return Real(0.5) * (
            v[IDX_EX] * v[IDX_EX] + v[IDX_EY] * v[IDX_EY] + v[IDX_EZ] * v[IDX_EZ]
        )

    def b_energy(v: Array[Real, MAXWELL_NC]) -> Real:
        return Real(0.5) * (
            v[IDX_BX] * v[IDX_BX] + v[IDX_BY] * v[IDX_BY] + v[IDX_BZ] * v[IDX_BZ]
        )

    var t_final = sqrt(Real(2.0)) / C_LIGHT
    var dt = spatial.suggest_dt(mesh)
    root_print("CFL dt =", dt)
    root_print("frame   time   ∫E_energy dV   ∫B_energy dV   ∫total dV")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_maxwell_frame(fw.next_path(t), mesh, q)
        var ee = mesh.integrate_fn(q, e_energy)
        var be = mesh.integrate_fn(q, b_energy)
        root_print(frame, t, ee, be, ee + be)

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until(
                ctx, q, t, t_final * Real(frame + 1) / Real(NUM_FRAMES - 1), dt,
                compute_rhs, no_op_adjust[2, P, MAXWELL_NC],
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/cavity.pvd")
