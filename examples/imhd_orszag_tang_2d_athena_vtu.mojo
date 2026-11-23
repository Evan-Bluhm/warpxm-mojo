# 2D MHD Orszag-Tang vortex — Athena code normalization — with VTU
# time-series output. Direct visual comparison target for the canonical
# Athena reference figures at:
#   https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/
#
# The setup mirrors `~/GitHub/Athena-Cversion/src/prob/orszag-tang.c`:
#   * Domain [0, 1]² periodic; γ = 5/3
#   * d_0 = 25/(36π), p_0 = 5/(12π), v_0 = 1, B_0 = 1/√(4π)
#   * Vector potential A_z = (B_0/4π) cos(4π x) + (B_0/2π) cos(2π y),
#     so B_x = -B_0 sin(2π y), B_y = B_0 sin(4π x).
#
# Frames are dumped from t = 0 to t = 4 at uniform intervals of 0.1 (the
# canonical Athena figures are at t = 0.5 and t = 1). The time step follows
# the CFL condition. Per frame the driver prints (∫ρ dV, ∫E dV, ∫ψ² dV)
# so the user can verify mass / energy conservation and that GLM
# divergence cleaning is keeping ∇·B errors damped.
#
#     pixi run example imhd_orszag_tang_2d_athena_vtu
#     # then open output/orszag_tang_athena/orszag_tang_athena.pvd in ParaView

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *


comptime P: Int = 3
comptime NX: Int = 128
comptime NY: Int = 128
comptime GAMMA: Real = 5.0 / 3.0
comptime V0: Real = 1.0
comptime C_H: Real = 3.0
comptime C_P_SQ: Real = 0.18
comptime T_FINAL: Real = 4.0
comptime NUM_FRAMES: Int = 41  # frames 0 .. 40 inclusive
comptime State = DGState[2, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    var d0: Real = Real(25.0) / (Real(36.0) * PI)
    var p0: Real = Real(5.0) / (Real(12.0) * PI)
    var B0: Real = Real(1.0) / sqrt(Real(4.0) * PI)

    def initial(x: Real, y: Real) {imm} -> Array[Real, MHD_GLM_NC]:
        var ux = -V0 * sin(TWO_PI * y)
        var uy = V0 * sin(TWO_PI * x)
        var bx = -B0 * sin(TWO_PI * y)
        var by = B0 * sin(Real(2.0) * TWO_PI * x)
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[IDX_RHO] = d0
        v[IDX_MX] = d0 * ux
        v[IDX_MY] = d0 * uy
        v[IDX_E] = (
            p0 / (GAMMA - Real(1.0)) + Real(0.5) * d0 * (ux * ux + uy * uy)
            + Real(0.5) * (bx * bx + by * by)
        )
        v[IDX_BX] = bx
        v[IDX_BY] = by
        return v^
    mesh.fill(q, initial)

    var app = ImhdGLM(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
        c_h=C_H,
        c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[2, P](app)
    var stepper = SSPRK3[2, P, MHD_GLM_NC](ctx, mesh.num_elements, mesh.num_ghost)
    # The limiter acts on deviations above α = alpha_coeff · dx^1.5. The
    # default coefficient (1000) gives α ≈ 0.7 here, above every density in
    # the problem (ρ ≤ 0.6), so the limiter would be idle and the run would
    # rest on the positivity floor until a node evacuates (around t = 0.5).
    # α ≈ 0.07 keeps the limiter working without flattening the extrema:
    # ρ ∈ [0.042, 0.45] at t = 0.3 against Athena's [0.047, 0.55].
    var limiter = MoeRossmanith[2, P, MHD_GLM_NC](
        ctx, mesh.num_elements, mesh.num_ghost, alpha_coeff=Real(100.0),
    )
    var positivity = PositivityMHD[2, P, MHD_GLM_NC](
        num_elements=mesh.num_elements,
        gamma=GAMMA,
        rho_eps=Real(1.0e-10),
        p_eps=Real(1.0e-10),
    )
    var fw = FrameWriter("orszag_tang_athena")

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    def psi_squared(v: Array[Real, MHD_GLM_NC]) -> Real:
        return v[IDX_PSI] * v[IDX_PSI]

    var mass_0 = mesh.integrate(q, IDX_RHO)
    var e_0 = mesh.integrate(q, IDX_E)

    root_print("Orszag-Tang vortex (Athena units) — domain [0,1]², P =", P, ", grid", NX, "x", NY)
    root_print("  d_0 =", d0, " p_0 =", p0, " B_0 =", B0)
    root_print("time   ∫ρ dV   ∫E dV   ∫ψ² dV   rel Δmass   rel ΔE   CFL dt")

    var t: Real = 0.0
    for frame in range(NUM_FRAMES):
        dump_imhd_glm_frame(fw.next_path(t), mesh, q, GAMMA)
        var mass = mesh.integrate(q, IDX_RHO)
        var e_tot = mesh.integrate(q, IDX_E)
        root_print(
            t, mass, e_tot, mesh.integrate_fn(q, psi_squared),
            (mass - mass_0) / mass_0, (e_tot - e_0) / e_0, suggest_dt(q),
        )

        if frame < NUM_FRAMES - 1:
            t = stepper.run_until_cfl(
                ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
                compute_rhs, adjust, suggest_dt, dt_interval=10,
            )

    fw.write_pvd()
    root_print("Wrote", NUM_FRAMES, "frames and", fw.directory + "/orszag_tang_athena.pvd")
