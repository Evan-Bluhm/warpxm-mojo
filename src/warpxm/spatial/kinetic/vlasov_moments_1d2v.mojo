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
# Parallelism: one block per x-element, `(P_x + 1) × MOMENT_LANES`
# threads. Each lane reduces a stride of v1-elements and the lanes are
# combined through shared memory (see `_moment_kernel_1d2v`).

from max.gpu import block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state import DGState


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


comptime MOMENT_ZEROTH: Int = 0
comptime MOMENT_FIRST_V1: Int = 1
comptime MOMENT_FIRST_V2: Int = 2
comptime MOMENT_SECOND: Int = 3

comptime MOMENT_LANES: Int = 32
"""Threads per x-node that share one velocity-space reduction."""


def _moment_kernel_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int, KIND: Int,
    QLayout: TensorLayout, OutLayout: TensorLayout,
    NodeV1Layout: TensorLayout, NodeV2Layout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    node_v1: TileTensor[WXM_DTYPE, NodeV1Layout, MutAnyOrigin],
    node_v2: TileTensor[WXM_DTYPE, NodeV2Layout, MutAnyOrigin],
    result: TileTensor[WXM_DTYPE, OutLayout, MutAnyOrigin],
    num_x_i64: Int64, num_v1_i64: Int64, num_v2_i64: Int64,
    jacobian_v1: Real, jacobian_v2: Real,
):
    """`result(x) = ∫∫ g(v1, v2) f(x, v1, v2) dv1 dv2` per x-node, with
    `g = 1, v1, v2, ½(v1² + v2²)` for `KIND = MOMENT_ZEROTH, _FIRST_V1,
    _FIRST_V2, _SECOND`.

    Thread layout: `block_idx.x = ex`, `(thread_idx.x, thread_idx.y) =
    (ix, lane)`. Lane `l` sums the v1-elements `l, l + LANES, …` (over
    all of v2), parks its partial sum in shared memory, and lane 0 adds
    the partials in lane order. The result is deterministic, and the
    serial depth per thread is `⌈num_v1 / LANES⌉ · num_v2` elements
    instead of `num_v1 · num_v2`: with one thread per x-node this
    reduction cost as much as the kinetic right-hand side.
    """
    var num_x = Int(num_x_i64)
    var num_v1 = Int(num_v1_i64)
    var num_v2 = Int(num_v2_i64)
    comptime P1x = P_x + 1
    comptime P1v1 = P_v1 + 1
    comptime P1v2 = P_v2 + 1
    comptime LANES = MOMENT_LANES
    comptime assert q.flat_rank == 1
    comptime assert node_v1.flat_rank == 1
    comptime assert node_v2.flat_rank == 1
    comptime assert result.flat_rank == 1
    comptime assert KIND >= 0 and KIND <= 3, "unknown moment kind"

    var ex = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    var lane = Int(thread_idx.y)

    var partial = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[P1x, LANES]())

    if ex >= num_x or ix >= P1x or lane >= LANES:
        barrier()
        return

    var acc: Real = Real(0.0)
    var ev1 = lane
    while ev1 < num_v1:
        comptime for iv1 in range(P1v1):
            comptime wv1: Real = NodalBasis1D[P_v1].weights()[iv1]
            var c1: Real = wv1 * jacobian_v1
            var v1: Real = rebind[Scalar[WXM_DTYPE]](
                node_v1[ev1 * P1v1 + iv1]
            )
            for ev2 in range(num_v2):
                comptime for iv2 in range(P1v2):
                    comptime wv2: Real = NodalBasis1D[P_v2].weights()[iv2]
                    var idx = _phase_idx_1d2v[P_x, P_v1, P_v2](
                        ex, ev1, ev2, iv2, iv1, ix, num_v1, num_v2,
                    )
                    var fv: Real = rebind[Scalar[WXM_DTYPE]](q[idx])
                    var w: Real = c1 * wv2 * jacobian_v2
                    comptime if KIND == MOMENT_ZEROTH:
                        acc += w * fv
                    comptime if KIND == MOMENT_FIRST_V1:
                        acc += w * v1 * fv
                    comptime if KIND == MOMENT_FIRST_V2:
                        var v2: Real = rebind[Scalar[WXM_DTYPE]](
                            node_v2[ev2 * P1v2 + iv2]
                        )
                        acc += w * v2 * fv
                    comptime if KIND == MOMENT_SECOND:
                        var v2: Real = rebind[Scalar[WXM_DTYPE]](
                            node_v2[ev2 * P1v2 + iv2]
                        )
                        acc += w * Real(0.5) * (v1 * v1 + v2 * v2) * fv
        ev1 += LANES

    partial[ix, lane] = rebind[partial.ElementType](acc)
    barrier()

    if lane == 0:
        var total: Real = Real(0.0)
        comptime for l in range(LANES):
            total += rebind[Scalar[WXM_DTYPE]](partial[ix, l])
        result[ex * P1x + ix] = rebind[result.ElementType](total)


struct VlasovMoments1D2V[P_x: Int, P_v1: Int, P_v2: Int]:
    """Host-side launcher for the 1D2V velocity-space moment kernels."""

    def __init__(out self):
        pass

    def _launch[KIND: Int](
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut result: DGState[1, Self.P_x, 1],
    ) raises:
        comptime P1x = Self.P_x + 1
        comptime P1v1 = Self.P_v1 + 1
        comptime P1v2 = Self.P_v2 + 1
        var q_layout = row_major(Int(q.buf_len))
        var out_layout = row_major(Int(result.buf_len))
        var node_v1_layout = row_major(Int(mesh.num_v1 * P1v1))
        var node_v2_layout = row_major(Int(mesh.num_v2 * P1v2))

        comptime kernel = _moment_kernel_1d2v[
            Self.P_x, Self.P_v1, Self.P_v2, KIND,
            type_of(q_layout), type_of(out_layout),
            type_of(node_v1_layout), type_of(node_v2_layout),
        ]
        ctx.enqueue_function[kernel](
            TileTensor(q.buf, q_layout),
            TileTensor(mesh.node_v1_buf, node_v1_layout),
            TileTensor(mesh.node_v2_buf, node_v2_layout),
            TileTensor(result.buf, out_layout),
            Int64(mesh.num_x), Int64(mesh.num_v1), Int64(mesh.num_v2),
            mesh.jacobian_v1, mesh.jacobian_v2,
            grid_dim=mesh.num_x,
            block_dim=(P1x, MOMENT_LANES),
        )

    def zeroth_moment(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut rho: DGState[1, Self.P_x, 1],
    ) raises:
        """Compute `ρ(x) = ∫∫ f dv1 dv2` into `rho`."""
        self._launch[MOMENT_ZEROTH](ctx, mesh, q, rho)

    def first_moment_v1(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut j: DGState[1, Self.P_x, 1],
    ) raises:
        """Compute `J_x(x) = ∫∫ v1 · f dv1 dv2` into `j`."""
        self._launch[MOMENT_FIRST_V1](ctx, mesh, q, j)

    def first_moment_v2(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut j: DGState[1, Self.P_x, 1],
    ) raises:
        """Compute `J_y(x) = ∫∫ v2 · f dv1 dv2` into `j`."""
        self._launch[MOMENT_FIRST_V2](ctx, mesh, q, j)

    def second_moment(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[
            Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC,
        ],
        mut e_dens: DGState[1, Self.P_x, 1],
    ) raises:
        """Compute `E_kin(x) = ½ ∫∫ (v1²+v2²) · f dv1 dv2` into `e_dens`."""
        self._launch[MOMENT_SECOND](ctx, mesh, q, e_dens)
