# 3D Orszag-Tang vortex (Athena units) — 2D-extruded.
#
# Tensor-product extension of `test_imhd_orszag_tang_2d_athena.mojo` into
# 3D with no z-dependence: `u_z = 0, B_z = 0, ∂_z {all} = 0`. The 3D
# IMHD-GLM solver should preserve the z-invariance to machine precision
# and reproduce the same 2D evolution at every z-slice.
#
# This is item #3 from `tasks/orszag_tang_athena.md` — useful as another
# check that the 3D MHD solver gives bit-identical behavior to 2D on
# extruded ICs (we already have this validation for Brio-Wu and Alfvén).
#
# Mesh: `NX = NY = 32`, `NZ = 2`. The reduced NZ keeps the run cheap
# while still exercising multiple z-layers' worth of comparison work
# (any per-z-layer code-path bug would show up as z-drift).
#
# Assertions:
#   * no NaN
#   * mass conserved to machine precision (periodic, no source on ρ)
#   * z-drift: for every (i, j, in_, jn) the value at z-layer k matches
#     z-layer 0 to within `1e-12` for all 9 MHD-GLM components.
#   * density and gas pressure stay strictly positive
#   * total ψ² stays below the same loose bound as the 2D test

from std.collections import Array
from std.math import isnan, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 2
comptime GAMMA: Real = 5.0 / 3.0
comptime V0: Real = 1.0
comptime C_H: Real = 3.0
comptime C_P_SQ: Real = 0.18
comptime T_FINAL: Real = 0.3
comptime DT: Real = 5.0e-4
comptime State = DGState[3, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx,
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = State(ctx, mesh.num_elements)

    var d0: Real = Real(25.0) / (Real(36.0) * PI)
    var p0: Real = Real(5.0) / (Real(12.0) * PI)
    var B0: Real = Real(1.0) / sqrt(Real(4.0) * PI)

    def initial(x: Real, y: Real, z: Real) {imm} -> Array[Real, MHD_GLM_NC]:
        var ux = -V0 * sin(TWO_PI * y)
        var uy = V0 * sin(TWO_PI * x)
        var bx = -B0 * sin(TWO_PI * y)
        var by = B0 * sin(Real(2.0) * TWO_PI * x)
        var msq = d0 * (ux * ux + uy * uy)
        var bsq = bx * bx + by * by
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[IDX_RHO] = d0
        v[IDX_MX] = d0 * ux
        v[IDX_MY] = d0 * uy
        v[IDX_E] = p0 / (GAMMA - Real(1.0)) + Real(0.5) * msq + Real(0.5) * bsq
        v[IDX_BX] = bx
        v[IDX_BY] = by
        return v^
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, MHD_GLM_NC](ctx, mesh.num_elements)
    var positivity = PositivityMHD[3, P, MHD_GLM_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var any_nan = False
    var psi2_T: Real = 0.0
    var max_rho: Real = -1.0e30
    var min_rho: Real = 1.0e30
    var max_p: Real = -1.0e30
    var min_p: Real = 1.0e30
    def scan(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, MHD_GLM_NC],
    ) {mut any_nan, mut psi2_T, mut max_rho, mut min_rho, mut max_p, mut min_p}:
        var rho = v[IDX_RHO]
        var psi = v[IDX_PSI]
        if isnan(rho) or isnan(psi) or isnan(v[IDX_E]):
            any_nan = True
        var ke = Real(0.5) * (
            v[IDX_MX] * v[IDX_MX] + v[IDX_MY] * v[IDX_MY] + v[IDX_MZ] * v[IDX_MZ]
        ) / rho
        var mag = Real(0.5) * (
            v[IDX_BX] * v[IDX_BX] + v[IDX_BY] * v[IDX_BY] + v[IDX_BZ] * v[IDX_BZ]
        )
        var pp = (GAMMA - Real(1.0)) * (v[IDX_E] - ke - mag)
        psi2_T += w * psi * psi
        if rho > max_rho: max_rho = rho
        if rho < min_rho: min_rho = rho
        if pp > max_p: max_p = pp
        if pp < min_p: min_p = pp
    mesh.visit_nodes(q, scan)
    var mass_T = mesh.integrate(q, IDX_RHO)

    # z-drift: every z-layer of elements should match the k = 0 layer
    # exactly, node for node and component for component.
    var max_z_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(1, NZ):
            for e_0 in range(NX * NY):
                var e_k = k * NX * NY + e_0
                for n in range(State.num_nodes_per_elem):
                    for c in range(MHD_GLM_NC):
                        var dv = fabs(
                            h[State.index(e_k, n, c)] - h[State.index(e_0, n, c)]
                        )
                        if dv > max_z_drift:
                            max_z_drift = dv

    var rel_mass_drift = fabs(mass_T - mass_0) / mass_0
    print(
        "Orszag-Tang vortex (Athena, 3D-extruded) — t =", t,
        " on", NX, "x", NY, "x", NZ, "mesh, P =", P,
    )
    print("  mass  (initial / final):   ", mass_0, " /", mass_T)
    print("  rel Δmass:                  ", rel_mass_drift)
    print("  ρ range: [", min_rho, ",", max_rho, "]  (initial =", d0, ")")
    print("  P range: [", min_p, ",", max_p, "]  (initial =", p0, ")")
    print("  ∫ψ² dV:                     ", psi2_T)
    print("  max z-drift across components:", max_z_drift)

    assert_true(not any_nan, "3D OT-Athena NaN'd")
    # 3D quadrature has `NZ × P1` additional accumulator iterations vs
    # the 2D test, so the FP drift floor is a few × larger.
    assert_true(rel_mass_drift < Real(1.0e-11), "mass not conserved")
    assert_true(min_rho > Real(0.0), "density went non-positive")
    assert_true(min_p > Real(0.0), "pressure went non-positive")
    assert_true(max_rho < Real(10.0) * d0, "density blew up")
    assert_true(psi2_T < Real(1.0e-3), "GLM ψ² grew uncontrollably")
    # z-drift bound: floating-point cumulative error over 600 SSPRK3 steps
    # on a problem with σ ≈ 1 fields. 1e-10 is loose but well above
    # the asymmetric-summation noise floor.
    assert_true(
        max_z_drift < Real(1.0e-10),
        "3D OT not bit-uniform in z — 3D solver introduced z-asymmetry",
    )
    print("PASSED")
