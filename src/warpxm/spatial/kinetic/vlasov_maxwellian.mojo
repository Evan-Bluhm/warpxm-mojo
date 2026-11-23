# Local-Maxwellian kernels for the generic kinetic stack: the BGK
# collision source and the 5-moment (Euler) → kinetic initialiser.
#
# Both evaluate, at every phase node, the Maxwellian
#
#     f_M(x, v) = n (2π θ)^(−VDIM/2) exp(−|v − u|² / (2θ))
#
# whose density `n`, drift `u` and thermal speed squared `θ` come from a
# five-component state on the companion `BlockMesh[XDIM, P_x]`:
#
#   * BGK (`VlasovBGKSource`): the velocity moments of `f` itself
#     (`VlasovMoments`), `u = Γ/n`, `θ = (2 W/n − |u|²) / VDIM`, and the
#     kernel adds `(f_M − f)/τ` to the right-hand side. `f_M` has the
#     density, momentum and energy of `f`, so the operator conserves them.
#     Mirrors upstream `wmapplication_bgk_collision_source.{h,cc}`.
#   * Initialiser (`apply_5moment_to_kinetic`): an Euler state
#     `(ρ, ρu, e)`, `n = ρ/m`, `u = (ρu)/(ρ m)`, `θ = p/(n m)` with
#     `p = (γ − 1)(e − ½|ρu|²/ρ)`, and the kernel overwrites `f` with
#     `f_M`. With `VDIM < 3` this is the marginal over the missing
#     velocity components. Mirrors upstream
#     `apps/vlasovmaxwell/5moment_to_kinetic.{h,cc}`.
#
# The work is pointwise, one thread per phase node.

from std.collections import Array
from std.math import ceildiv, exp
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_E
from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, PI
from warpxm.mesh.phase_mesh import PhaseMesh, PhaseInts
from warpxm.spatial.dg_launch import RealTensor
from warpxm.spatial.kinetic.vlasov_moments import (
    VlasovMoments, VLASOV_MOMENT_NC, MOM_DENSITY, MOM_FLUX_X, MOM_ENERGY,
)
from warpxm.variable.phase_state import PhaseState
from warpxm.variable.state import DGState


comptime MAXWELLIAN_FLOOR: Real = 1.0e-12
"""Floor on the density and on the thermal speed squared, which keeps the
Maxwellian finite where the moments are empty or (through round-off)
give a non-positive temperature."""

comptime _MODE_BGK: Int = 0
comptime _MODE_FROM_FLUID: Int = 1
comptime _BLOCK: Int = 256


@always_inline
def _maxwellian_at[
    XDIM: Int, VDIM: Int, P_x: Int, P_v: Int, MODE: Int,
](
    source: RealTensor,
    node_v: RealTensor,
    counts: PhaseInts,
    t: Int,
    p0: Real,
    p1: Real,
) -> Real:
    """`f_M` at flat phase node `t`. `source` is the five-component state
    per configuration node: the moments of `f` (BGK) or an Euler state
    with `(p0, p1) = (mass, γ)` (initialiser)."""
    comptime ND = XDIM + VDIM
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime NXN = P1x ** XDIM
    comptime NVN = P1v ** VDIM
    comptime NPE = NXN * NVN
    comptime NS = 5
    comptime assert VLASOV_MOMENT_NC == NS and EULER_NC == NS
    comptime assert VLASOV_NC == 1

    var n = Array[Int, ND](fill=1)
    comptime for a in range(ND):
        n[a] = Int(counts[a])
    var num_v_elements = 1
    comptime for d in range(VDIM):
        num_v_elements *= n[XDIM + d]

    var phase_e = t // NPE
    var local_n = t - phase_e * NPE
    var v_node = local_n // NXN
    var xn = local_n - v_node * NXN
    var xe_phase = UInt32(phase_e) // UInt32(num_v_elements)
    var ve = UInt32(phase_e) - xe_phase * UInt32(num_v_elements)

    # Companion-mesh element (x innermost there, last axis innermost here).
    var xe = Array[Int, XDIM](fill=0)
    var rem_x = xe_phase
    comptime for k in range(XDIM - 1):
        comptime a = XDIM - 1 - k
        var n_a = UInt32(n[a])
        var quo = rem_x // n_a
        xe[a] = Int(rem_x - quo * n_a)
        rem_x = quo
    xe[0] = Int(rem_x)
    var xe_block = 0
    var x_stride = 1
    comptime for a in range(XDIM):
        xe_block += xe[a] * x_stride
        x_stride *= n[a]

    # Velocity element per axis (last axis innermost).
    var ev = Array[Int, VDIM](fill=0)
    var rem_v = ve
    comptime for k in range(VDIM - 1):
        comptime d = VDIM - 1 - k
        var n_d = UInt32(n[XDIM + d])
        var quo = rem_v // n_d
        ev[d] = Int(rem_v - quo * n_d)
        rem_v = quo
    ev[0] = Int(rem_v)

    # Density, drift and thermal speed squared at this configuration node.
    var sb = (xe_block * NXN + xn) * NS
    var s0 = rebind[Real](source[sb])
    var s4 = rebind[Real](source[sb + 4])
    var density: Real
    var theta: Real
    var drift_scale: Real
    comptime if MODE == _MODE_BGK:
        var n_safe = s0 if s0 > MAXWELLIAN_FLOOR else MAXWELLIAN_FLOOR
        var inv_n = Real(1.0) / n_safe
        var u_sq = Real(0.0)
        comptime for d in range(VDIM):
            var u_d = rebind[Real](source[sb + MOM_FLUX_X + d]) * inv_n
            u_sq += u_d * u_d
        comptime INV_VDIM: Real = 1.0 / Real(VDIM)
        density = s0
        drift_scale = inv_n
        theta = (Real(2.0) * s4 * inv_n - u_sq) * INV_VDIM
    else:
        var mass = p0
        var gamma = p1
        var rho = s0 if s0 > MAXWELLIAN_FLOOR else MAXWELLIAN_FLOOR
        var inv_rho = Real(1.0) / rho
        var inv_mass = Real(1.0) / mass
        var m_sq = Real(0.0)
        comptime for c in range(3):
            var m_c = rebind[Real](source[sb + IDX_MX + c])
            m_sq += m_c * m_c
        var pressure = (gamma - Real(1.0)) * (s4 - Real(0.5) * m_sq * inv_rho)
        density = rho * inv_mass
        drift_scale = inv_rho * inv_mass
        # θ = T / m with T = p / n.
        theta = pressure * inv_rho
    if theta < MAXWELLIAN_FLOOR:
        theta = MAXWELLIAN_FLOOR

    var w_sq = Real(0.0)
    var v_offset = 0
    var rem_node = v_node
    comptime for d in range(VDIM):
        var j_d = rem_node % P1v
        rem_node = rem_node // P1v
        var v_d = rebind[Real](node_v[v_offset + ev[d] * P1v + j_d])
        var w_d = v_d - rebind[Real](source[sb + 1 + d]) * drift_scale
        w_sq += w_d * w_d
        v_offset += n[XDIM + d] * P1v

    var inv_theta = Real(1.0) / theta
    comptime INV_TWO_PI: Real = 1.0 / (2.0 * PI)
    var k = INV_TWO_PI * inv_theta
    var norm: Real
    comptime match VDIM:
        case 1:
            norm = sqrt(k)
        case 2:
            norm = k
        case _:
            norm = k * sqrt(k)
    return density * norm * exp(Real(-0.5) * w_sq * inv_theta)


def _bgk_kernel[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int](
    moments: RealTensor,
    q: RealTensor,
    rhs: RealTensor,
    node_v: RealTensor,
    counts: PhaseInts,
    num_nodes_i64: Int64,
    inv_tau: Real,
):
    """`rhs += (f_M − q) / τ`, one thread per phase node."""
    var t = Int(global_idx.x)
    if t >= Int(num_nodes_i64):
        return
    var f_m = _maxwellian_at[XDIM, VDIM, P_x, P_v, _MODE_BGK](
        moments, node_v, counts, t, Real(0.0), Real(0.0),
    )
    var f = rebind[Real](q[t])
    var r = rebind[Real](rhs[t])
    rhs[t] = rebind[rhs.ElementType](r + (f_m - f) * inv_tau)


def _from_fluid_kernel[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int](
    fluid: RealTensor,
    q: RealTensor,
    node_v: RealTensor,
    counts: PhaseInts,
    num_nodes_i64: Int64,
    mass: Real,
    gamma: Real,
):
    """`q = f_M` of the Euler state, one thread per phase node."""
    var t = Int(global_idx.x)
    if t >= Int(num_nodes_i64):
        return
    q[t] = rebind[q.ElementType](
        _maxwellian_at[XDIM, VDIM, P_x, P_v, _MODE_FROM_FLUID](
            fluid, node_v, counts, t, mass, gamma,
        )
    )


struct VlasovBGKSource[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int]:
    """BGK collision operator `(f_M − f)/τ` on a generic phase mesh. Owns
    the moment launcher and the moment state it needs."""

    comptime Phase = PhaseState[
        Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, VLASOV_NC,
    ]

    var moments: VlasovMoments[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v]
    var mom: DGState[Self.XDIM, Self.P_x, VLASOV_MOMENT_NC]

    def __init__(
        out self,
        var ctx: DeviceContext,
        mesh: PhaseMesh[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v],
    ) raises:
        self.moments = VlasovMoments[
            Self.XDIM, Self.VDIM, Self.P_x, Self.P_v,
        ](ctx, mesh)
        self.mom = DGState[Self.XDIM, Self.P_x, VLASOV_MOMENT_NC](
            ctx, mesh.num_x_elements,
        )

    def apply(
        mut self,
        var ctx: DeviceContext,
        mesh: PhaseMesh[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v],
        mut q: Self.Phase,
        mut rhs: Self.Phase,
        tau: Real,
    ) raises:
        """Add `(f_M − f)/τ` to `rhs`, `f_M` being the Maxwellian with
        the density, momentum and energy of `q` at each configuration
        node. Call it after the solver's `compute_rhs`."""
        self.moments.compute(ctx, mesh, q, self.mom)
        var q_layout = row_major(Int(q.buf_len))
        comptime kernel = _bgk_kernel[
            Self.XDIM, Self.VDIM, Self.P_x, Self.P_v,
        ]
        ctx.enqueue_function[kernel](
            TileTensor(self.mom.buf, row_major(Int(self.mom.buf_len))),
            TileTensor(q.buf, q_layout),
            TileTensor(rhs.buf, q_layout),
            TileTensor(mesh.node_v_buf, row_major(Int(mesh.node_v_len))),
            mesh.kernel_counts(), Int64(q.total_nodes),
            Real(1.0) / tau,
            grid_dim=ceildiv(q.total_nodes, _BLOCK), block_dim=_BLOCK,
        )


def apply_5moment_to_kinetic[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int](
    var ctx: DeviceContext,
    mesh: PhaseMesh[XDIM, VDIM, P_x, P_v],
    mut fluid: DGState[XDIM, P_x, EULER_NC],
    mut q: PhaseState[XDIM, VDIM, P_x, P_v, VLASOV_NC],
    mass: Real,
    gamma: Real,
) raises:
    """Overwrite `q` with the Maxwellian that has the density, velocity
    and temperature of the Euler state `fluid` at each configuration
    node (the first `VDIM` velocity components)."""
    if (
        fluid.num_elements != mesh.num_x_elements
        or q.num_phase_elements != mesh.num_phase_elements
    ):
        raise Error("apply_5moment_to_kinetic: state sizes do not match the mesh")
    var q_layout = row_major(Int(q.buf_len))
    comptime kernel = _from_fluid_kernel[XDIM, VDIM, P_x, P_v]
    ctx.enqueue_function[kernel](
        TileTensor(fluid.buf, row_major(Int(fluid.buf_len))),
        TileTensor(q.buf, q_layout),
        TileTensor(mesh.node_v_buf, row_major(Int(mesh.node_v_len))),
        mesh.kernel_counts(), Int64(q.total_nodes),
        mass, gamma,
        grid_dim=ceildiv(q.total_nodes, _BLOCK), block_dim=_BLOCK,
    )
