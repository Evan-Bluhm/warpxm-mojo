# Velocity-space moments of a 1D2V Vlasov distribution.
#
# Tensor-product extension of `vlasov_moments_1d1v.mojo`. Each kernel
# reduces the phase-space buffer along both velocity axes to produce a
# per-x-node scalar field:
#
#   ρ(x)      = ∫∫ f(x, v1, v2) dv1 dv2
#   J_x(x)    = ∫∫ v1 · f(x, v1, v2) dv1 dv2
#   J_y(x)    = ∫∫ v2 · f(x, v1, v2) dv1 dv2
#   E_kin(x)  = ½ ∫∫ (v1² + v2²) · f(x, v1, v2) dv1 dv2
#
# Parallelism: one thread per output (ex, ix). Each thread serially
# scans every `(ev1, iv1, ev2, iv2)` tuple. Same complexity scaling as
# the 1D1V variant — adequate at current 1D2V phase-space sizes; switch
# to a block-level tree reduction if benchmarks later show this is hot.

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state_1d import DGState1D


@always_inline
def _phase_idx_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int,
](
    ex: Int, ev1: Int, ev2: Int,
    iv2: Int, iv1: Int, ix: Int,
    num_v1: Int, num_v2: Int,
) -> Int:
    """Flat buffer index for a 1D2V phase node. Mirrors the storage
    convention in `variable/phase_state_1d2v.mojo` for NC=1."""
    comptime P1x = P_x + 1
    comptime P1v1 = P_v1 + 1
    comptime P1v2 = P_v2 + 1
    comptime nodes_per_phase = P1x * P1v1 * P1v2
    var phase_e = (ex * num_v1 + ev1) * num_v2 + ev2
    var local_n = (iv2 * P1v1 + iv1) * P1x + ix
    return (phase_e * nodes_per_phase + local_n) * VLASOV_NC


def _zeroth_moment_kernel_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int,
    QLayout: TensorLayout, RhoLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rho: TileTensor[WXM_DTYPE, RhoLayout, MutAnyOrigin],
    num_x: Int, num_v1: Int, num_v2: Int,
    jacobian_v1: Real, jacobian_v2: Real,
):
    """Compute `ρ(x) = ∫∫ f(x, v1, v2) dv1 dv2` per x-node via tensor-
    product GLL quadrature. Thread layout: `block_idx.x = ex`,
    `thread_idx.x = ix`."""
    comptime P1x = P_x + 1
    comptime P1v1 = P_v1 + 1
    comptime P1v2 = P_v2 + 1
    comptime assert q.flat_rank == 1
    comptime assert rho.flat_rank == 1

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    if ex >= num_x or ix >= P1x:
        return

    var w_v1 = NodalBasis1D[P_v1].weights()
    var w_v2 = NodalBasis1D[P_v2].weights()
    var acc: Real = Real(0.0)
    for ev1 in range(num_v1):
        for ev2 in range(num_v2):
            comptime for iv2 in range(P1v2):
                var wv2: Real = w_v2[iv2]
                comptime for iv1 in range(P1v1):
                    var wv1: Real = w_v1[iv1]
                    var idx = _phase_idx_1d2v[P_x, P_v1, P_v2](
                        ex, ev1, ev2, iv2, iv1, ix, num_v1, num_v2,
                    )
                    var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
                    acc += wv1 * jacobian_v1 * wv2 * jacobian_v2 * fv

    var out_idx = ex * P1x + ix
    rho[out_idx] = rebind[rho.ElementType](acc)


def _first_moment_v1_kernel_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int,
    QLayout: TensorLayout, JLayout: TensorLayout,
    NodeV1Layout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    node_v1: TileTensor[WXM_DTYPE, NodeV1Layout, MutAnyOrigin],
    j: TileTensor[WXM_DTYPE, JLayout, MutAnyOrigin],
    num_x: Int, num_v1: Int, num_v2: Int,
    jacobian_v1: Real, jacobian_v2: Real,
):
    """Compute `J_x(x) = ∫∫ v1 · f(x, v1, v2) dv1 dv2` per x-node."""
    comptime P1x = P_x + 1
    comptime P1v1 = P_v1 + 1
    comptime P1v2 = P_v2 + 1
    comptime assert q.flat_rank == 1
    comptime assert node_v1.flat_rank == 1
    comptime assert j.flat_rank == 1

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    if ex >= num_x or ix >= P1x:
        return

    var w_v1 = NodalBasis1D[P_v1].weights()
    var w_v2 = NodalBasis1D[P_v2].weights()
    var acc: Real = Real(0.0)
    for ev1 in range(num_v1):
        comptime for iv1 in range(P1v1):
            var wv1: Real = w_v1[iv1]
            var v1: Real = rebind[Scalar[WXM_DTYPE]](
                node_v1[ev1 * P1v1 + iv1]
            )
            for ev2 in range(num_v2):
                comptime for iv2 in range(P1v2):
                    var wv2: Real = w_v2[iv2]
                    var idx = _phase_idx_1d2v[P_x, P_v1, P_v2](
                        ex, ev1, ev2, iv2, iv1, ix, num_v1, num_v2,
                    )
                    var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
                    acc += (
                        wv1 * jacobian_v1 * wv2 * jacobian_v2 * v1 * fv
                    )

    var out_idx = ex * P1x + ix
    j[out_idx] = rebind[j.ElementType](acc)


def _first_moment_v2_kernel_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int,
    QLayout: TensorLayout, JLayout: TensorLayout,
    NodeV2Layout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    node_v2: TileTensor[WXM_DTYPE, NodeV2Layout, MutAnyOrigin],
    j: TileTensor[WXM_DTYPE, JLayout, MutAnyOrigin],
    num_x: Int, num_v1: Int, num_v2: Int,
    jacobian_v1: Real, jacobian_v2: Real,
):
    """Compute `J_y(x) = ∫∫ v2 · f(x, v1, v2) dv1 dv2` per x-node — the
    second component of the species current density that only appears in
    1D2V (the 1D1V kernel had a single `v` axis)."""
    comptime P1x = P_x + 1
    comptime P1v1 = P_v1 + 1
    comptime P1v2 = P_v2 + 1
    comptime assert q.flat_rank == 1
    comptime assert node_v2.flat_rank == 1
    comptime assert j.flat_rank == 1

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    if ex >= num_x or ix >= P1x:
        return

    var w_v1 = NodalBasis1D[P_v1].weights()
    var w_v2 = NodalBasis1D[P_v2].weights()
    var acc: Real = Real(0.0)
    for ev2 in range(num_v2):
        comptime for iv2 in range(P1v2):
            var wv2: Real = w_v2[iv2]
            var v2: Real = rebind[Scalar[WXM_DTYPE]](
                node_v2[ev2 * P1v2 + iv2]
            )
            for ev1 in range(num_v1):
                comptime for iv1 in range(P1v1):
                    var wv1: Real = w_v1[iv1]
                    var idx = _phase_idx_1d2v[P_x, P_v1, P_v2](
                        ex, ev1, ev2, iv2, iv1, ix, num_v1, num_v2,
                    )
                    var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
                    acc += (
                        wv1 * jacobian_v1 * wv2 * jacobian_v2 * v2 * fv
                    )

    var out_idx = ex * P1x + ix
    j[out_idx] = rebind[j.ElementType](acc)


def _second_moment_kernel_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int,
    QLayout: TensorLayout, ELayout: TensorLayout,
    NodeV1Layout: TensorLayout, NodeV2Layout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    node_v1: TileTensor[WXM_DTYPE, NodeV1Layout, MutAnyOrigin],
    node_v2: TileTensor[WXM_DTYPE, NodeV2Layout, MutAnyOrigin],
    e_dens: TileTensor[WXM_DTYPE, ELayout, MutAnyOrigin],
    num_x: Int, num_v1: Int, num_v2: Int,
    jacobian_v1: Real, jacobian_v2: Real,
):
    """Compute `E_kin(x) = ½ ∫∫ (v1² + v2²) · f(x, v1, v2) dv1 dv2` per
    x-node. With ρ and the two first moments this gives the bulk
    velocity `u = J / ρ` and the temperature
    `T = (2 E_kin / ρ) − |u|²` once `ρ` and `J` are also known."""
    comptime P1x = P_x + 1
    comptime P1v1 = P_v1 + 1
    comptime P1v2 = P_v2 + 1
    comptime assert q.flat_rank == 1
    comptime assert node_v1.flat_rank == 1
    comptime assert node_v2.flat_rank == 1
    comptime assert e_dens.flat_rank == 1

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    if ex >= num_x or ix >= P1x:
        return

    var w_v1 = NodalBasis1D[P_v1].weights()
    var w_v2 = NodalBasis1D[P_v2].weights()
    var acc: Real = Real(0.0)
    for ev1 in range(num_v1):
        comptime for iv1 in range(P1v1):
            var wv1: Real = w_v1[iv1]
            var v1: Real = rebind[Scalar[WXM_DTYPE]](
                node_v1[ev1 * P1v1 + iv1]
            )
            for ev2 in range(num_v2):
                comptime for iv2 in range(P1v2):
                    var wv2: Real = w_v2[iv2]
                    var v2: Real = rebind[Scalar[WXM_DTYPE]](
                        node_v2[ev2 * P1v2 + iv2]
                    )
                    var idx = _phase_idx_1d2v[P_x, P_v1, P_v2](
                        ex, ev1, ev2, iv2, iv1, ix, num_v1, num_v2,
                    )
                    var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
                    acc += (
                        wv1 * jacobian_v1 * wv2 * jacobian_v2
                        * Real(0.5) * (v1 * v1 + v2 * v2) * fv
                    )

    var out_idx = ex * P1x + ix
    e_dens[out_idx] = rebind[e_dens.ElementType](acc)


struct VlasovMoments1D2V[P_x: Int, P_v1: Int, P_v2: Int]:
    """Host-side launcher for the 1D2V velocity-space moment kernels."""

    def __init__(out self):
        pass

    def zeroth_moment(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut rho: DGState1D[Self.P_x, 1],
    ) raises:
        """Compute `ρ(x) = ∫∫ f dv1 dv2` into `rho`."""
        comptime P1x = Self.P_x + 1
        var q_layout = row_major(Idx(q.buf_len))
        var rho_layout = row_major(Idx(rho.buf_len))

        var q_view = TileTensor(q.buf, q_layout)
        var rho_view = TileTensor(rho.buf, rho_layout)

        comptime kernel = _zeroth_moment_kernel_1d2v[
            Self.P_x, Self.P_v1, Self.P_v2,
            type_of(q_layout), type_of(rho_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rho_view,
            mesh.num_x, mesh.num_v1, mesh.num_v2,
            mesh.jacobian_v1, mesh.jacobian_v2,
            grid_dim=mesh.num_x,
            block_dim=P1x,
        )

    def first_moment_v1(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut j: DGState1D[Self.P_x, 1],
    ) raises:
        """Compute `J_x(x) = ∫∫ v1 · f dv1 dv2` into `j`."""
        comptime P1x = Self.P_x + 1
        comptime P1v1 = Self.P_v1 + 1
        var q_layout = row_major(Idx(q.buf_len))
        var j_layout = row_major(Idx(j.buf_len))
        var node_v1_layout = row_major(Idx(mesh.num_v1 * P1v1))

        var q_view = TileTensor(q.buf, q_layout)
        var node_v1_view = TileTensor(mesh.node_v1_buf, node_v1_layout)
        var j_view = TileTensor(j.buf, j_layout)

        comptime kernel = _first_moment_v1_kernel_1d2v[
            Self.P_x, Self.P_v1, Self.P_v2,
            type_of(q_layout), type_of(j_layout), type_of(node_v1_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, node_v1_view, j_view,
            mesh.num_x, mesh.num_v1, mesh.num_v2,
            mesh.jacobian_v1, mesh.jacobian_v2,
            grid_dim=mesh.num_x,
            block_dim=P1x,
        )

    def first_moment_v2(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut j: DGState1D[Self.P_x, 1],
    ) raises:
        """Compute `J_y(x) = ∫∫ v2 · f dv1 dv2` into `j`."""
        comptime P1x = Self.P_x + 1
        comptime P1v2 = Self.P_v2 + 1
        var q_layout = row_major(Idx(q.buf_len))
        var j_layout = row_major(Idx(j.buf_len))
        var node_v2_layout = row_major(Idx(mesh.num_v2 * P1v2))

        var q_view = TileTensor(q.buf, q_layout)
        var node_v2_view = TileTensor(mesh.node_v2_buf, node_v2_layout)
        var j_view = TileTensor(j.buf, j_layout)

        comptime kernel = _first_moment_v2_kernel_1d2v[
            Self.P_x, Self.P_v1, Self.P_v2,
            type_of(q_layout), type_of(j_layout), type_of(node_v2_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, node_v2_view, j_view,
            mesh.num_x, mesh.num_v1, mesh.num_v2,
            mesh.jacobian_v1, mesh.jacobian_v2,
            grid_dim=mesh.num_x,
            block_dim=P1x,
        )

    def second_moment(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut e_dens: DGState1D[Self.P_x, 1],
    ) raises:
        """Compute `E_kin(x) = ½ ∫∫ (v1²+v2²) · f dv1 dv2` into `e_dens`."""
        comptime P1x = Self.P_x + 1
        comptime P1v1 = Self.P_v1 + 1
        comptime P1v2 = Self.P_v2 + 1
        var q_layout = row_major(Idx(q.buf_len))
        var e_layout = row_major(Idx(e_dens.buf_len))
        var node_v1_layout = row_major(Idx(mesh.num_v1 * P1v1))
        var node_v2_layout = row_major(Idx(mesh.num_v2 * P1v2))

        var q_view = TileTensor(q.buf, q_layout)
        var node_v1_view = TileTensor(mesh.node_v1_buf, node_v1_layout)
        var node_v2_view = TileTensor(mesh.node_v2_buf, node_v2_layout)
        var e_view = TileTensor(e_dens.buf, e_layout)

        comptime kernel = _second_moment_kernel_1d2v[
            Self.P_x, Self.P_v1, Self.P_v2,
            type_of(q_layout), type_of(e_layout),
            type_of(node_v1_layout), type_of(node_v2_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, node_v1_view, node_v2_view, e_view,
            mesh.num_x, mesh.num_v1, mesh.num_v2,
            mesh.jacobian_v1, mesh.jacobian_v2,
            grid_dim=mesh.num_x,
            block_dim=P1x,
        )
