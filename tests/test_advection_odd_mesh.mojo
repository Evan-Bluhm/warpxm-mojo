# Advection on meshes whose element count is not a multiple of the DG
# core's elements-per-block packing (`DG_CORE_ELEMS_PER_BLOCK = 4`).
#
# The last block of such a mesh is only partly filled. The threads past
# the last element still have to meet the others at the kernel's barrier:
# an early return there deadlocks the launch (the run never finishes), and
# a wrong element index corrupts or skips the last elements. Each case
# advects a sine wave for a tenth of a period and checks
#
#   * the largest nodal error against the exact solution. An element that
#     was skipped would be off by the distance the wave moved (about 0.5
#     here), far above the discretisation error;
#   * conservation of the integral of a second, offset tracer.
#
# Cases at P = 2 and P = 3 in 1D, 2D and 3D, with element counts that
# leave 1, 2 and 3 elements in the last block.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *
from warpxm.spatial.dg_core import DG_CORE_ELEMS_PER_BLOCK


comptime NC: Int = 2
comptime T_END: Real = 0.1


def _check(
    label: String, num_elements: Int, linf: Real, tol: Real,
    mass_0: Real, mass_1: Real,
) raises:
    print(
        label, ": elements", num_elements, " max error", linf,
        " |Δ∫q|", fabs(mass_1 - mass_0),
    )
    assert_true(
        num_elements % DG_CORE_ELEMS_PER_BLOCK != 0,
        label + ": the mesh must leave a partial block",
    )
    assert_true(linf < tol, label + ": nodal error above the discretisation level")
    assert_true(
        fabs(mass_1 - mass_0) < Real(1.0e-12),
        label + ": tracer integral not conserved",
    )


def _run_1d[P: Int](var ctx: DeviceContext, n: Int, tol: Real) raises:
    comptime State = DGState[1, P, NC]
    var mesh = BlockMesh[1, P](ctx, Real(0.0), Real(1.0), n, True)
    var q = State(ctx, mesh.num_elements)
    def initial(x: Real) -> Array[Real, NC]:
        return [sin(TWO_PI * x), Real(2.0) + sin(TWO_PI * x)]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, 1)

    var spatial = DGAdvectionSolver[1, P, NC](
        Advection[NC](velocity_x=Real(1.0), cfl_max=Real(0.5)),
    )
    var stepper = SSPRK3[1, P, NC](ctx, mesh.num_elements)
    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
    _ = stepper.run_until(
        ctx, q, Real(0.0), T_END, spatial.suggest_dt(mesh), compute_rhs,
        no_op_adjust[1, P, NC],
    )

    var linf: Real = 0.0
    def error(x: Real, w: Real, v: Array[Real, NC]) {mut linf}:
        linf = fmax(linf, fabs(v[0] - sin(TWO_PI * (x - T_END))))
    mesh.visit_nodes(q, error)
    _check(
        "1D P=" + String(P), mesh.num_elements, linf, tol,
        mass_0, mesh.integrate(q, 1),
    )


def _run_2d[P: Int](var ctx: DeviceContext, nx: Int, ny: Int, tol: Real) raises:
    comptime State = DGState[2, P, NC]
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), nx, ny, True, True,
    )
    var q = State(ctx, mesh.num_elements)
    def initial(x: Real, y: Real) -> Array[Real, NC]:
        var s = sin(TWO_PI * x) * sin(TWO_PI * y)
        return [s, Real(2.0) + s]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, 1)

    var spatial = DGAdvectionSolver[2, P, NC](
        Advection[NC](velocity_x=Real(1.0), velocity_y=Real(-0.5), cfl_max=Real(0.5)),
    )
    var stepper = SSPRK3[2, P, NC](ctx, mesh.num_elements)
    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
    _ = stepper.run_until(
        ctx, q, Real(0.0), T_END, spatial.suggest_dt(mesh), compute_rhs,
        no_op_adjust[2, P, NC],
    )

    var linf: Real = 0.0
    def error(x: Real, y: Real, w: Real, v: Array[Real, NC]) {mut linf}:
        var exact = (
            sin(TWO_PI * (x - T_END)) * sin(TWO_PI * (y + Real(0.5) * T_END))
        )
        linf = fmax(linf, fabs(v[0] - exact))
    mesh.visit_nodes(q, error)
    _check(
        "2D P=" + String(P), mesh.num_elements, linf, tol,
        mass_0, mesh.integrate(q, 1),
    )


def _run_3d[P: Int](
    var ctx: DeviceContext, nx: Int, ny: Int, nz: Int, tol: Real,
) raises:
    comptime State = DGState[3, P, NC]
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx, ny, nz,
    )
    var q = State(ctx, mesh.num_elements)
    def initial(x: Real, y: Real, z: Real) -> Array[Real, NC]:
        var s = sin(TWO_PI * x) * sin(TWO_PI * y) * sin(TWO_PI * z)
        return [s, Real(2.0) + s]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, 1)

    var spatial = DGAdvectionSolver[3, P, NC](
        Advection[NC](
            velocity_x=Real(1.0), velocity_y=Real(-0.5), velocity_z=Real(0.25),
            cfl_max=Real(0.5),
        ),
    )
    var stepper = SSPRK3[3, P, NC](ctx, mesh.num_elements)
    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
    _ = stepper.run_until(
        ctx, q, Real(0.0), T_END, spatial.suggest_dt(mesh), compute_rhs,
        no_op_adjust[3, P, NC],
    )

    var linf: Real = 0.0
    def error(x: Real, y: Real, z: Real, w: Real, v: Array[Real, NC]) {mut linf}:
        var exact = (
            sin(TWO_PI * (x - T_END))
            * sin(TWO_PI * (y + Real(0.5) * T_END))
            * sin(TWO_PI * (z - Real(0.25) * T_END))
        )
        linf = fmax(linf, fabs(v[0] - exact))
    mesh.visit_nodes(q, error)
    _check(
        "3D P=" + String(P), mesh.num_elements, linf, tol,
        mass_0, mesh.integrate(q, 1),
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    _run_1d[2](ctx, 37, Real(4.0e-4))
    _run_1d[3](ctx, 38, Real(5.0e-6))
    _run_2d[2](ctx, 13, 11, Real(2.0e-2))
    _run_2d[3](ctx, 9, 6, Real(6.0e-3))
    _run_3d[2](ctx, 9, 5, 5, Real(0.12))
    _run_3d[3](ctx, 7, 3, 3, Real(5.0e-2))

    print("PASSED")
