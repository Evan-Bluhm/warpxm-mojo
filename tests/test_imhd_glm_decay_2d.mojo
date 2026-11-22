# MixedGLM divergence-cleaning regression.
#
# IC: a localised Gaussian bump in B_x on a uniform, quiescent background.
# The bump has a non-zero ∇·B in its periphery; this should excite ψ
# through the hyperbolic coupling (`∂ψ/∂t + c_h²·∇·B = ...`), after which
# the parabolic damping term decays ψ exponentially.
#
# Validation:
#   1) The system stays bounded — no NaN, no runaway growth.
#   2) `∫ ψ² dV` first rises (charging) and then decays toward zero.
#   3) `∫ B_x² dV` stays bounded — the localised B-bump propagates and
#      diffuses but does not grow.
#
# Background: ρ = 1, u = 0, P = 1, γ = 5/3, B_y = B_z = 0.
# Bump:       B_x = δ · exp(−r²/σ²) with r centred at the domain centre.

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime DELTA: Real = 0.1
comptime SIGMA: Real = 0.1
comptime C_H: Real = 1.0
comptime C_P_SQ: Real = 0.05   # strong damping ⇒ short decay time
comptime DT: Real = 5.0e-3
comptime State = DGState[2, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, MHD_GLM_NC]:
        var r2 = (x - Real(0.5)) * (x - Real(0.5)) + (y - Real(0.5)) * (y - Real(0.5))
        var bx_bump = DELTA * exp(-r2 / (SIGMA * SIGMA))
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[IDX_RHO] = 1.0
        v[IDX_E] = Real(1.0) / (GAMMA - Real(1.0)) + Real(0.5) * bx_bump * bx_bump
        v[IDX_BX] = bx_bump
        return v^
    mesh.fill(q, initial)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[2, P](app)
    var stepper = SSPRK3[2, P, MHD_GLM_NC](ctx, mesh.num_elements)

    # Compute ∫f² dV over a single component via inline GLL quadrature.

    def psi_sq(v: Array[Real, MHD_GLM_NC]) -> Real:
        return v[IDX_PSI] * v[IDX_PSI]

    def bx_sq(v: Array[Real, MHD_GLM_NC]) -> Real:
        return v[IDX_BX] * v[IDX_BX]

    var psi2_0 = mesh.integrate_fn(q, psi_sq)
    var bx2_0 = mesh.integrate_fn(q, bx_sq)
    print("t = 0     : ∫ψ²  =", psi2_0, "  ∫B_x² =", bx2_0)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    var t: Real = 0.0
    var t_charge: Real = 0.5
    var t_final: Real = 2.0
    while t < t_charge - Real(0.5) * DT:
        stepper.step(ctx, q, DT, compute_rhs, no_op_adjust[2, P, MHD_GLM_NC])
        t += DT
    ctx.synchronize()
    var psi2_max = mesh.integrate_fn(q, psi_sq)
    print("t =", t, ": ∫ψ²  =", psi2_max, "  (charging done)")

    while t < t_final - Real(0.5) * DT:
        stepper.step(ctx, q, DT, compute_rhs, no_op_adjust[2, P, MHD_GLM_NC])
        t += DT
    ctx.synchronize()

    var psi2_end = mesh.integrate_fn(q, psi_sq)
    var bx2_end = mesh.integrate_fn(q, bx_sq)
    print("t =", t, ": ∫ψ²  =", psi2_end, "  ∫B_x² =", bx2_end)
    print("ψ² decay ratio (end / max):", psi2_end / psi2_max)

    assert_true(psi2_max > psi2_0 + Real(1.0e-8),
                "ψ should pick up energy from ∇·B coupling")
    assert_true(psi2_end < Real(0.02) * psi2_max,
                "ψ should decay at least 98 % from its peak under GLM damping")
    assert_true(bx2_end < Real(2.0) * bx2_0,
                "B_x² should stay bounded — no runaway growth")
    print("PASSED")
