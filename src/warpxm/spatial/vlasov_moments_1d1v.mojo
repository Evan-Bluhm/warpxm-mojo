# Velocity-space moments of a 1D1V Vlasov distribution.
#
# Mirrors upstream's `wmapplication_zeroth_moment.{h,cc}` (and its
# first / second-moment siblings). Each moment kernel reduces the
# phase-space buffer along the v-axis to produce a per-x-node scalar
# field:
#
#   ρ(x)   = ∫ f(x, v) dv
#   nV(x)  = ∫ v · f(x, v) dv         (first moment / momentum density)
#   nT(x)  = ∫ ½ (v − u)² · f(x, v) dv   (energy density centered on the bulk velocity)
#
# This slice ships the **zeroth moment** only. The first and second
# moments will follow the same per-thread reduction pattern when the
# Maxwell-coupling slice needs them.
#
# Parallelism: one thread per output x-node. Each thread serially scans
# every velocity element and every v-node within it, summing
# `w_iv · jacobian_v · f(ex, ev, ix, iv)`. For 1D1V phase-space sizes
# this is amply parallel; if benchmarks later show the v-direction
# reduction is a bottleneck, switch to a tree reduction over (ev, iv).

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


def _zeroth_moment_kernel_1d1v[
    P_x: Int, P_v: Int,
    QLayout: TensorLayout, RhoLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rho: TileTensor[WXM_DTYPE, RhoLayout, MutAnyOrigin],
    num_x: Int, num_v: Int,
    jacobian_v: Real,
):
    """Compute `ρ(x) = ∫ f(x, v) dv` per x-node via GLL quadrature
    along v. One thread per (ex, ix) output node.

    Thread layout: `block_idx.x = ex`, `thread_idx.x = ix`."""
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime assert q.flat_rank == 1
    comptime assert rho.flat_rank == 1

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    if ex >= num_x or ix >= P1x:
        return

    var weights = NodalBasis1D[P_v].weights()
    var acc: Real = Real(0.0)
    for ev in range(num_v):
        var phase_e = ex * num_v + ev
        comptime for iv in range(P1v):
            var w_iv: Real = weights[iv]
            var idx = (
                (phase_e * P1v + iv) * P1x + ix
            ) * VLASOV_NC
            var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
            acc += w_iv * jacobian_v * fv

    var out_idx = ex * P1x + ix
    rho[out_idx] = rebind[rho.ElementType](acc)


def _first_moment_kernel_1d1v[
    P_x: Int, P_v: Int,
    QLayout: TensorLayout, JLayout: TensorLayout,
    NodeVLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    node_v: TileTensor[WXM_DTYPE, NodeVLayout, MutAnyOrigin],
    j: TileTensor[WXM_DTYPE, JLayout, MutAnyOrigin],
    num_x: Int, num_v: Int,
    jacobian_v: Real,
):
    """Compute `J(x) = ∫ v · f(x, v) dv` per x-node — the species
    current density / momentum density. One thread per (ex, ix) output
    node, same reduction pattern as the zeroth moment but with the
    extra `v_iv` factor read from `mesh.node_v_buf`.

    Thread layout: `block_idx.x = ex`, `thread_idx.x = ix`."""
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime assert q.flat_rank == 1
    comptime assert node_v.flat_rank == 1
    comptime assert j.flat_rank == 1

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    if ex >= num_x or ix >= P1x:
        return

    var weights = NodalBasis1D[P_v].weights()
    var acc: Real = Real(0.0)
    for ev in range(num_v):
        var phase_e = ex * num_v + ev
        comptime for iv in range(P1v):
            var w_iv: Real = weights[iv]
            var v_iv: Real = rebind[Scalar[WXM_DTYPE]](
                node_v[ev * P1v + iv]
            )
            var idx = (
                (phase_e * P1v + iv) * P1x + ix
            ) * VLASOV_NC
            var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
            acc += w_iv * jacobian_v * v_iv * fv

    var out_idx = ex * P1x + ix
    j[out_idx] = rebind[j.ElementType](acc)


def _second_moment_kernel_1d1v[
    P_x: Int, P_v: Int,
    QLayout: TensorLayout, ELayout: TensorLayout,
    NodeVLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    node_v: TileTensor[WXM_DTYPE, NodeVLayout, MutAnyOrigin],
    e_dens: TileTensor[WXM_DTYPE, ELayout, MutAnyOrigin],
    num_x: Int, num_v: Int,
    jacobian_v: Real,
):
    """Compute `E_kin(x) = ½ ∫ v² · f(x, v) dv` per x-node — the
    species energy density. Same reduction as the first moment but
    with `½ · v²` in place of `v`. Useful as a diagnostic for
    temperature and for the kinetic-energy half of the Vlasov-Maxwell
    energy budget.

    Thread layout: `block_idx.x = ex`, `thread_idx.x = ix`."""
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime assert q.flat_rank == 1
    comptime assert node_v.flat_rank == 1
    comptime assert e_dens.flat_rank == 1

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    if ex >= num_x or ix >= P1x:
        return

    var weights = NodalBasis1D[P_v].weights()
    var acc: Real = Real(0.0)
    for ev in range(num_v):
        var phase_e = ex * num_v + ev
        comptime for iv in range(P1v):
            var w_iv: Real = weights[iv]
            var v_iv: Real = rebind[Scalar[WXM_DTYPE]](
                node_v[ev * P1v + iv]
            )
            var idx = (
                (phase_e * P1v + iv) * P1x + ix
            ) * VLASOV_NC
            var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
            acc += w_iv * jacobian_v * Real(0.5) * v_iv * v_iv * fv

    var out_idx = ex * P1x + ix
    e_dens[out_idx] = rebind[e_dens.ElementType](acc)


struct VlasovMoments1D1V[P_x: Int, P_v: Int]:
    """Host-side launcher for the 1D1V velocity-space moment kernels."""

    def __init__(out self):
        pass

    def zeroth_moment(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D1V[Self.P_x, Self.P_v],
        mut q: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        mut rho: DGState1D[Self.P_x, 1],
    ) raises:
        """Compute `ρ(x) = ∫ f(x, v) dv` into `rho`."""
        comptime P1x = Self.P_x + 1
        var q_layout = row_major(Idx(q.buf_len))
        var rho_layout = row_major(Idx(rho.buf_len))

        var q_view = TileTensor(q.buf, q_layout)
        var rho_view = TileTensor(rho.buf, rho_layout)

        comptime kernel = _zeroth_moment_kernel_1d1v[
            Self.P_x, Self.P_v, type_of(q_layout), type_of(rho_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rho_view, mesh.num_x, mesh.num_v, mesh.jacobian_v,
            grid_dim=mesh.num_x,
            block_dim=P1x,
        )

    def first_moment(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D1V[Self.P_x, Self.P_v],
        mut q: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        mut j: DGState1D[Self.P_x, 1],
    ) raises:
        """Compute `J(x) = ∫ v · f(x, v) dv` into `j` — the species
        current density / momentum density."""
        comptime P1x = Self.P_x + 1
        comptime P1v = Self.P_v + 1
        var q_layout = row_major(Idx(q.buf_len))
        var j_layout = row_major(Idx(j.buf_len))
        var node_v_layout = row_major(Idx(mesh.num_v * P1v))

        var q_view = TileTensor(q.buf, q_layout)
        var node_v_view = TileTensor(mesh.node_v_buf, node_v_layout)
        var j_view = TileTensor(j.buf, j_layout)

        comptime kernel = _first_moment_kernel_1d1v[
            Self.P_x, Self.P_v,
            type_of(q_layout), type_of(j_layout), type_of(node_v_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, node_v_view, j_view,
            mesh.num_x, mesh.num_v, mesh.jacobian_v,
            grid_dim=mesh.num_x,
            block_dim=P1x,
        )

    def second_moment(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D1V[Self.P_x, Self.P_v],
        mut q: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        mut e_dens: DGState1D[Self.P_x, 1],
    ) raises:
        """Compute `E_kin(x) = ½ ∫ v² · f(x, v) dv` into `e_dens` —
        the species kinetic energy density. Combined with the zeroth
        moment ρ, this gives temperature `T = (2 E_kin / ρ) − u²`
        once a bulk velocity `u` is also known."""
        comptime P1x = Self.P_x + 1
        comptime P1v = Self.P_v + 1
        var q_layout = row_major(Idx(q.buf_len))
        var e_layout = row_major(Idx(e_dens.buf_len))
        var node_v_layout = row_major(Idx(mesh.num_v * P1v))

        var q_view = TileTensor(q.buf, q_layout)
        var node_v_view = TileTensor(mesh.node_v_buf, node_v_layout)
        var e_view = TileTensor(e_dens.buf, e_layout)

        comptime kernel = _second_moment_kernel_1d1v[
            Self.P_x, Self.P_v,
            type_of(q_layout), type_of(e_layout), type_of(node_v_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, node_v_view, e_view,
            mesh.num_x, mesh.num_v, mesh.jacobian_v,
            grid_dim=mesh.num_x,
            block_dim=P1x,
        )
