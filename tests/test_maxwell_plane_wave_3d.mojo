# 3D Maxwell plane-wave regression.
#
# IC: a TM wave propagating along x̂ with E in ẑ, B in ŷ:
#     E_z(x, y, z, 0) =  sin(2π x)
#     B_y(x, y, z, 0) = -sin(2π x) / c
#     all other components = 0
#
# With c = 1 the wave traverses one wavelength in T = 1 and returns to IC.
# Although the physics is 1D, we run on a 3D mesh — the 3D kernel must
# handle the extra ζ-axis cleanly:
#   * the y and z flux divergences should produce zero net contribution
#     (fields are uniform in those directions)
#   * E_x, E_y, B_x, B_z must remain zero throughout

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 8
comptime NZ: Int = 8
comptime C_LIGHT: Real = 1.0
comptime DT: Real = 5.0e-4
comptime State = DGState[3, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, MAXWELL_NC]:
        var s = sin(TWO_PI * x)
        return [Real(0.0), Real(0.0), s, Real(0.0), -s / C_LIGHT, Real(0.0)]
    mesh.fill(q, initial)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver[3, P](app)
    var stepper = SSPRK3[3, P, MAXWELL_NC](ctx, mesh.num_elements)

    var t_final: Real = 1.0 / C_LIGHT

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t = stepper.run_until(
        ctx, q, Real(0.0), t_final, DT, compute_rhs,
        no_op_adjust[3, P, MAXWELL_NC],
    )
    ctx.synchronize()

    def exact_ez(x: Real, y: Real, z: Real) -> Real:
        return sin(TWO_PI * x)
    def exact_by(x: Real, y: Real, z: Real) -> Real:
        return -sin(TWO_PI * x) / C_LIGHT
    var l2_Ez = mesh.l2_error(q, IDX_EZ, exact_ez)
    var l2_By = mesh.l2_error(q, IDX_BY, exact_by)
    # E_x, E_y, B_x, B_z must remain zero.
    var max_other: Real = 0.0
    def scan(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, MAXWELL_NC],
    ) {mut max_other}:
        for idx in [IDX_EX, IDX_EY, IDX_BX, IDX_BZ]:
            max_other = fmax(max_other, fabs(v[idx]))
    mesh.visit_nodes(q, scan)
    print("3D Maxwell plane wave — t =", t)
    print("  L2 E_z error:", l2_Ez)
    print("  L2 B_y error:", l2_By)
    print("  max |E_x, E_y, B_x, B_z| (should be ~0):", max_other)
    assert_true(l2_Ez < Real(1.0e-2), "E_z error too large")
    assert_true(l2_By < Real(1.0e-2), "B_y error too large")
    assert_true(max_other < Real(1.0e-10),
                "transverse components grew — 3D kernel has spurious coupling")
    print("PASSED")
