# The dimension-generic Vlasov solver against the hand-written 1D1V and
# 1D2V solvers.
#
# `DGVlasovSolver[XDIM, VDIM, P_x, P_v]` is a different kernel from
# `DGVlasovSolver1D1V` / `DGVlasovSolver1D2V` (no face tables, only `f` in
# shared memory, upwinding written as max / min of the speed), but it
# discretises the same equation the same way, so on the same data the two
# right-hand sides must agree to round-off. The data are a non-separable
# `f` and fields that vary in x, so every volume and surface term and both
# signs of every speed are exercised, on a periodic and on a copy-out
# configuration boundary.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_generic_rhs_1d.mojo

from std.collections import Array
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime NX: Int = 6
comptime NV: Int = 5
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MAX: Real = 3.0


def fields(x: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EX] = 0.7 * sin(TWO_PI * x) + 0.2
    v[IDX_EY] = -0.4 * cos(TWO_PI * x) + 0.1
    v[IDX_EZ] = 0.3 * sin(TWO_PI * x)
    v[IDX_BX] = 0.25
    v[IDX_BY] = 0.5 * cos(TWO_PI * x)
    v[IDX_BZ] = 0.6 * sin(TWO_PI * x) - 0.3
    return v^


def f_2d(x: Real, v1: Real) -> Real:
    return (1.0 + 0.3 * sin(TWO_PI * x + 0.4 * v1)) * exp(-0.5 * (v1 - 0.3) * (v1 - 0.3))


def f_3d(x: Real, v1: Real, v2: Real) -> Real:
    return (
        (1.0 + 0.3 * sin(TWO_PI * x + 0.4 * v1 - 0.2 * v2))
        * exp(-0.5 * (v1 - 0.3) * (v1 - 0.3) - 0.4 * (v2 + 0.2) * (v2 + 0.2) + 0.1 * v1 * v2)
    )


def _compare(
    label: String, mut a: DeviceBuffer[Real.dtype], mut b: DeviceBuffer[Real.dtype],
    size: Int,
) raises:
    var diff: Real = 0.0
    var scale: Real = 0.0
    with a.map_to_host() as ha:
        with b.map_to_host() as hb:
            for k in range(size):
                diff = fmax(diff, fabs(ha[k] - hb[k]))
                scale = fmax(scale, fabs(hb[k]))
    print(label, " max |rhs| =", scale, " max difference =", diff)
    assert_true(scale > Real(1.0), label + ": reference RHS is trivial")
    assert_true(
        diff < Real(1.0e-12) * scale, label + ": generic RHS differs from legacy",
    )


def check_1d1v[P_X: Int, P_V: Int](var ctx: DeviceContext, periodic: Bool) raises:
    var app = VlasovMaxwell(charge_mass_ratio=Real(-1.3), skin_depth_norm=Real(0.8))
    var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=periodic)
    var em = DGState[1, P_X, MAXWELL_NC](ctx, NX)
    x_mesh.fill(em, fields)

    var old_mesh = PhaseMesh1D1V[P_X, P_V](
        ctx, X_MIN, X_MAX, -V_MAX, V_MAX, num_x=NX, num_v=NV,
        periodic_x=periodic,
    )
    var old_q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    var old_rhs = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    old_mesh.fill(old_q, f_2d)
    DGVlasovSolver1D1V[P_X, P_V](app).compute_rhs(ctx, old_mesh, old_q, em, old_rhs)

    var mesh = PhaseMesh[1, 1, P_X, P_V](
        ctx, lo=[X_MIN, -V_MAX], hi=[X_MAX, V_MAX], n=[NX, NV],
        periodic=[periodic],
    )
    var q = PhaseState[1, 1, P_X, P_V, VLASOV_NC](ctx, mesh.num_phase_elements)
    var rhs = PhaseState[1, 1, P_X, P_V, VLASOV_NC](ctx, mesh.num_phase_elements)
    mesh.fill(q, f_2d)
    DGVlasovSolver[1, 1, P_X, P_V](app).compute_rhs(ctx, mesh, q, em, rhs)
    ctx.synchronize()

    assert_true(q.buf_len == old_q.buf_len, "1D1V layouts differ")
    _compare(
        "1D1V P_x=" + String(P_X) + " P_v=" + String(P_V)
        + (" periodic" if periodic else " copy-out"),
        rhs.buf, old_rhs.buf, q.buf_len,
    )


def check_1d2v[P_X: Int, P_V: Int](var ctx: DeviceContext, periodic: Bool) raises:
    var app = VlasovMaxwell(charge_mass_ratio=Real(-1.3), skin_depth_norm=Real(0.8))
    var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=periodic)
    var em = DGState[1, P_X, MAXWELL_NC](ctx, NX)
    x_mesh.fill(em, fields)

    var old_mesh = PhaseMesh1D2V[P_X, P_V, P_V](
        ctx, X_MIN, X_MAX, -V_MAX, V_MAX, -V_MAX, V_MAX,
        num_x=NX, num_v1=NV, num_v2=NV + 1, periodic_x=periodic,
    )
    var old_q = PhaseState1D2V[P_X, P_V, P_V, VLASOV_NC](ctx, NX, NV, NV + 1)
    var old_rhs = PhaseState1D2V[P_X, P_V, P_V, VLASOV_NC](ctx, NX, NV, NV + 1)
    old_mesh.fill(old_q, f_3d)
    DGVlasovSolver1D2V[P_X, P_V, P_V](app).compute_rhs(
        ctx, old_mesh, old_q, em, old_rhs,
    )

    var mesh = PhaseMesh[1, 2, P_X, P_V](
        ctx, lo=[X_MIN, -V_MAX, -V_MAX], hi=[X_MAX, V_MAX, V_MAX],
        n=[NX, NV, NV + 1], periodic=[periodic],
    )
    var q = PhaseState[1, 2, P_X, P_V, VLASOV_NC](ctx, mesh.num_phase_elements)
    var rhs = PhaseState[1, 2, P_X, P_V, VLASOV_NC](ctx, mesh.num_phase_elements)
    mesh.fill(q, f_3d)
    DGVlasovSolver[1, 2, P_X, P_V](app).compute_rhs(ctx, mesh, q, em, rhs)
    ctx.synchronize()

    assert_true(q.buf_len == old_q.buf_len, "1D2V layouts differ")
    _compare(
        "1D2V P_x=" + String(P_X) + " P_v=" + String(P_V)
        + (" periodic" if periodic else " copy-out"),
        rhs.buf, old_rhs.buf, q.buf_len,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    check_1d1v[2, 3](ctx, True)
    check_1d1v[3, 2](ctx, False)
    check_1d2v[2, 2](ctx, True)
    check_1d2v[1, 3](ctx, False)
    check_1d2v[3, 2](ctx, False)
    print("PASSED")
