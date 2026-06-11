# 5-moment Euler ⇒ 1D1V Vlasov Maxwellian initialiser.
#
# Mirrors upstream `apps/vlasovmaxwell/5moment_to_kinetic.{h,cc}`.
# Given an Euler 5-moment state `q_fluid = (ρ, ρu_x, ρu_y, ρu_z, e)`
# per physical x-node, fills a phase-state buffer with the local
# Maxwellian carrying the same (mass-normalised) moments at each
# x-node:
#
#     n    = ρ / m,
#     V_x  = (ρu_x) / (ρ · m)       (= u_x / m),
#     nT   = (γ − 1) · ( e − ½·|ρu|²/ρ )    [pressure],
#     T    = nT / n,
#     V_th = √(T / m).
#
# In 1D1V the marginal Maxwellian over `v_x` is
#
#     f_M(v_x) = n / √(2π V_th²) · exp( −(v_x − V_x)² / (2 V_th²) ).
#
# The other two velocity dimensions (`v_y`, `v_z`) are implicitly
# integrated out — upstream's `f_maxwellian` produces the same shape
# in 1D1V because the exp factors for unrepresented components are
# absorbed into the normalisation power
# `(1 / (2π V_th²))^(d_v / 2)`.
#
# This is a one-shot *initialiser*, not a time-step source: it
# overwrites the phase-state at each x-node with the reconstructed
# Maxwellian. Useful for kicking off a Vlasov simulation from an
# equilibrated Euler state, or for hybrid fluid/kinetic IC's.

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import exp
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, WXM_DTYPE, PI
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


comptime T_MIN: Real = 1.0e-12


def _5moment_to_kinetic_kernel_1d1v[
    P_x: Int, P_v: Int,
    FluidLayout: TensorLayout, QLayout: TensorLayout,
    NodeVLayout: TensorLayout,
](
    fluid: TileTensor[WXM_DTYPE, FluidLayout, MutAnyOrigin],
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    node_v: TileTensor[WXM_DTYPE, NodeVLayout, MutAnyOrigin],
    num_v: Int,
    mass: Real,
    gamma: Real,
    num_phase_elements: Int,
):
    """Per-phase-node Maxwellian projection. Overwrites `q` (does not
    accumulate).

    Thread layout: `block_idx.x = phase_e`,
    `(thread_idx.x, thread_idx.y) = (ix, iv)`."""
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime nodes_per_phase = P1x * P1v
    comptime NC = VLASOV_NC
    comptime assert fluid.flat_rank == 1
    comptime assert q.flat_rank == 1
    comptime assert node_v.flat_rank == 1

    var phase_e = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    var iv = Int(thread_idx.y)
    if phase_e >= num_phase_elements or ix >= P1x or iv >= P1v:
        return

    var ex = phase_e // num_v
    var ev = phase_e % num_v
    var local_n = iv * P1x + ix

    # Read fluid state at this x-node.
    var node_x = ex * P1x + ix
    var fb = node_x * EULER_NC
    var rho: Real = rebind[Scalar[WXM_DTYPE]](fluid[fb + IDX_RHO])
    var mx: Real = rebind[Scalar[WXM_DTYPE]](fluid[fb + IDX_MX])
    var my: Real = rebind[Scalar[WXM_DTYPE]](fluid[fb + IDX_MY])
    var mz: Real = rebind[Scalar[WXM_DTYPE]](fluid[fb + IDX_MZ])
    var e: Real = rebind[Scalar[WXM_DTYPE]](fluid[fb + IDX_E])

    # Reconstruct Maxwellian parameters. Floor `rho` to avoid div-by-zero
    # in pathological IC's; the resulting Maxwellian is effectively a
    # delta at the velocity-grid scale.
    var rho_safe = rho if rho > T_MIN else T_MIN
    var n = rho_safe / mass
    var v_bulk_x = mx / (rho_safe * mass)
    var nt = (gamma - Real(1.0)) * (
        e - Real(0.5) * (mx * mx + my * my + mz * mz) / rho_safe
    )
    var t = nt / n
    var t_safe = t if t > T_MIN else T_MIN
    var v_th_sq = t_safe / mass

    # Maxwellian at this phase node.
    var v_iv: Real = rebind[Scalar[WXM_DTYPE]](node_v[ev * P1v + iv])
    var dv_shift = v_iv - v_bulk_x
    var f_m = n / sqrt(Real(2.0) * PI * v_th_sq) * exp(
        -dv_shift * dv_shift / (Real(2.0) * v_th_sq)
    )

    # Overwrite the phase-state.
    var base = (phase_e * nodes_per_phase + local_n) * NC
    q[base] = rebind[q.ElementType](f_m)


def apply_5moment_to_kinetic_1d1v[P_x: Int, P_v: Int](
    var ctx: DeviceContext,
    mut mesh: PhaseMesh1D1V[P_x, P_v],
    mut fluid: DGState1D[P_x, EULER_NC],
    mut q: PhaseState1D1V[P_x, P_v, VLASOV_NC],
    mass: Real,
    gamma: Real,
) raises:
    """Populate `q` with the local Maxwellian carrying the moments of
    the 5-moment Euler state `fluid` at each x-node. Overwrites `q`."""
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    var fluid_layout = row_major(Idx(fluid.buf_len))
    var q_layout = row_major(Idx(q.buf_len))
    var node_v_layout = row_major(Idx(mesh.num_v * P1v))

    var fluid_view = TileTensor(fluid.buf, fluid_layout)
    var q_view = TileTensor(q.buf, q_layout)
    var node_v_view = TileTensor(mesh.node_v_buf, node_v_layout)

    comptime kernel = _5moment_to_kinetic_kernel_1d1v[
        P_x, P_v,
        type_of(fluid_layout), type_of(q_layout), type_of(node_v_layout),
    ]
    ctx.enqueue_function[kernel](
        fluid_view, q_view, node_v_view,
        mesh.num_v, mass, gamma, mesh.num_phase_elements,
        grid_dim=mesh.num_phase_elements,
        block_dim=(P1x, P1v),
    )
