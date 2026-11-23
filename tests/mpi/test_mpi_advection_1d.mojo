# Multi-rank regression: 1D advection through copy-out boundaries.
#
# A pulse leaves through the right boundary. Only the first and the last
# rank touch a physical boundary; the faces in between are partition
# faces. The final field must not depend on the number of ranks.

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *

comptime P: Int = 3
comptime State = DGState[1, P, 1]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), 37, False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real) -> Array[Real, 1]:
        var d = (x - Real(0.6)) / Real(0.08)
        return [exp(-d * d)]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, 0)

    var spatial = DGAdvectionSolver[1, P, 1](
        Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3[1, P, 1](ctx, mesh.num_elements, mesh.num_ghost)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(mesh)

    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), Real(0.8), compute_rhs, no_op_adjust[1, P, 1],
        suggest_dt,
    )
    var mass_1 = mesh.integrate(q, 0)
    var sum = mesh.checksum(q)
    if is_root():
        print("ranks =", num_ranks(), " mass", mass_0, "->", mass_1)
        print("CHECKSUM", sum)
    if not (mass_1 < Real(1.0e-3) * mass_0):
        raise Error("the pulse did not leave the domain")
    if is_root():
        print("PASSED")
