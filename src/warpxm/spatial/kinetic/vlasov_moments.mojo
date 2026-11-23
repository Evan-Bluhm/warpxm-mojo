# Velocity-space moments of a Vlasov distribution, generic over the
# phase-space dimensions.
#
# One kernel pass reduces `f` over all velocity axes and writes, per
# configuration-space node, the `VLASOV_MOMENT_NC` moments
#
#   MOM_DENSITY   n      = ∫ f dv
#   MOM_FLUX_X    n u_x  = ∫ v_1 f dv
#   MOM_FLUX_Y    n u_y  = ∫ v_2 f dv      (0 when VDIM < 2)
#   MOM_FLUX_Z    n u_z  = ∫ v_3 f dv      (0 when VDIM < 3)
#   MOM_ENERGY    ½ ∫ |v|² f dv
#
# into a `DGState[XDIM, P_x, VLASOV_MOMENT_NC]` on the companion
# `BlockMesh[XDIM, P_x]`. All the moments a coupled step needs (the three
# current components) come out of a single pass over `f` rather than one
# launch each.
#
# Parallelism. The velocity elements of one configuration element are
# split into chunks, and one block of `(P_x + 1)^XDIM × LANES` threads
# reduces one chunk: lane `l` sums the chunk's elements `l, l + LANES, …`,
# parks its partial sums in shared memory, and lane 0 adds them in lane
# order. The chunk size is chosen so that a thread walks at most
# `MOMENT_ELEMS_PER_THREAD` elements, so the thread count grows with the
# velocity mesh and a mesh with few configuration elements and many
# velocity elements still fills the device. With more than one chunk the
# per-chunk sums go to a small scratch buffer and a second, pointwise
# kernel adds them in chunk order. Either way the summation order is
# fixed, so the result is deterministic.

from std.collections import Array
from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from layout import TileTensor, row_major, stack_allocation

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.phase_mesh import PhaseMesh, PhaseInts
from warpxm.spatial.dg_launch import RealTensor
from warpxm.variable.phase_state import PhaseState
from warpxm.variable.state import DGState


comptime VLASOV_MOMENT_NC: Int = 5
comptime MOM_DENSITY: Int = 0
comptime MOM_FLUX_X: Int = 1
comptime MOM_FLUX_Y: Int = 2
comptime MOM_FLUX_Z: Int = 3
comptime MOM_ENERGY: Int = 4

comptime MOMENT_ELEMS_PER_THREAD: Int = 16
"""Most velocity elements one thread reduces serially (see the header)."""

comptime _SUM_BLOCK: Int = 256


def _moment_lanes(num_x_nodes: Int) -> Int:
    """Threads per configuration node sharing one velocity reduction: 32,
    or fewer when 32 per node would not fit in a 1024-thread block."""
    return 32 if num_x_nodes * 32 <= 1024 else 1024 // num_x_nodes


def _v_node_weight[P_v: Int, VDIM: Int](v_node: Int) -> Real:
    """Tensor-product GLL weight of velocity node `v_node` (compile time)."""
    var w: Real = 1.0
    var rem = v_node
    for _ in range(VDIM):
        w = w * NodalBasis1D[P_v].weights()[rem % (P_v + 1)]
        rem = rem // (P_v + 1)
    return w


def _vlasov_moments_kernel[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int](
    q: RealTensor,
    node_v: RealTensor,
    sums: RealTensor,
    counts: PhaseInts,
    chunking: PhaseInts,
    jacobian_v: Real,
):
    """Velocity moments of one chunk of velocity elements, per
    configuration node (see the file header). `chunking` is `(elements
    per chunk, number of chunks)`; `sums` is indexed `[x node, chunk,
    moment]`, which for a single chunk is the layout of the moment state
    itself. `jacobian_v` is the product of the velocity-axis Jacobians.

    Thread layout: `block_idx.x` is the configuration element in phase
    ordering, `block_idx.y` the chunk, `(thread_idx.x, thread_idx.y) =
    (x node, lane)`. The block is exactly `(P_x + 1)^XDIM × LANES`, so
    every thread reaches the barrier.
    """
    comptime ND = XDIM + VDIM
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime NXN = P1x ** XDIM
    comptime NVN = P1v ** VDIM
    comptime NPE = NXN * NVN
    comptime NM = VLASOV_MOMENT_NC
    comptime LANES = _moment_lanes(NXN)
    comptime assert VLASOV_NC == 1, "moments are taken of the scalar f"

    var n = Array[Int, ND](fill=1)
    comptime for a in range(ND):
        n[a] = Int(counts[a])
    var num_v_elements = 1
    comptime for d in range(VDIM):
        num_v_elements *= n[XDIM + d]

    var chunk_size = Int(chunking[0])
    var num_chunks = Int(chunking[1])
    var xe_phase = Int(block_idx.x)
    var chunk = Int(block_idx.y)
    var xn = Int(thread_idx.x)
    var lane = Int(thread_idx.y)

    # Companion-mesh element: the phase mesh numbers the configuration
    # axes last-innermost, `BlockMesh` first-innermost.
    var xe = Array[Int, XDIM](fill=0)
    var rem_x = UInt32(xe_phase)
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

    var partial = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[NXN, LANES, NM]())

    var acc = Array[Real, NM](fill=Real(0.0))
    var ve = chunk * chunk_size + lane
    var ve_end = (chunk + 1) * chunk_size
    if ve_end > num_v_elements:
        ve_end = num_v_elements
    while ve < ve_end:
        # Velocity element index along each axis (last axis innermost).
        var ev = Array[Int, VDIM](fill=0)
        var rem_v = UInt32(ve)
        comptime for k in range(VDIM - 1):
            comptime d = VDIM - 1 - k
            var n_d = UInt32(n[XDIM + d])
            var quo = rem_v // n_d
            ev[d] = Int(rem_v - quo * n_d)
            rem_v = quo
        ev[0] = Int(rem_v)

        # Node coordinates of this velocity element, per axis.
        var vv = Array[Real, VDIM * P1v](fill=Real(0.0))
        var v_offset = 0
        comptime for d in range(VDIM):
            comptime for j in range(P1v):
                vv[d * P1v + j] = rebind[Real](
                    node_v[v_offset + ev[d] * P1v + j]
                )
            v_offset += n[XDIM + d] * P1v

        var base = (xe_phase * num_v_elements + ve) * NPE + xn
        comptime for v_node in range(NVN):
            comptime w: Real = _v_node_weight[P_v, VDIM](v_node)
            var wf = w * rebind[Real](q[base + v_node * NXN])
            acc[MOM_DENSITY] += wf
            var v_sq = Real(0.0)
            comptime for d in range(VDIM):
                comptime j = (v_node // (P1v ** d)) % P1v
                var v_d = vv[d * P1v + j]
                acc[MOM_FLUX_X + d] += wf * v_d
                v_sq += v_d * v_d
            acc[MOM_ENERGY] += wf * v_sq
        ve += LANES

    comptime for m in range(NM):
        partial[xn, lane, m] = rebind[partial.ElementType](acc[m])
    barrier()

    if lane == 0:
        var out_base = ((xe_block * NXN + xn) * num_chunks + chunk) * NM
        comptime for m in range(NM):
            var total = Real(0.0)
            comptime for l in range(LANES):
                total += rebind[Real](partial[xn, l, m])
            comptime scale: Real = 0.5 if m == MOM_ENERGY else 1.0
            sums[out_base + m] = rebind[sums.ElementType](
                total * (jacobian_v * scale)
            )


def _sum_chunks_kernel(
    partials: RealTensor,
    moments: RealTensor,
    num_x_nodes_i64: Int64,
    num_chunks_i64: Int64,
):
    """`moments[x node, m] = Σ_chunk partials[x node, chunk, m]`, one
    thread per configuration node, chunks added in order."""
    comptime NM = VLASOV_MOMENT_NC
    var node = Int(global_idx.x)
    if node >= Int(num_x_nodes_i64):
        return
    var num_chunks = Int(num_chunks_i64)
    var total = Array[Real, NM](fill=Real(0.0))
    for c in range(num_chunks):
        comptime for m in range(NM):
            total[m] += rebind[Real](partials[(node * num_chunks + c) * NM + m])
    comptime for m in range(NM):
        moments[node * NM + m] = rebind[moments.ElementType](total[m])


struct VlasovMoments[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int]:
    """Host-side launcher for the velocity-moment kernels. Owns the
    per-chunk scratch buffer, so build it once per mesh and reuse it."""

    var chunk_size: Int
    var num_chunks: Int
    var partials_len: Int
    var partials: DeviceBuffer[Real.dtype]
    """`[x node, chunk, moment]` sums, used when `num_chunks > 1`."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        mesh: PhaseMesh[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v],
    ) raises:
        comptime NXN = (Self.P_x + 1) ** Self.XDIM
        self.chunk_size = _moment_lanes(NXN) * MOMENT_ELEMS_PER_THREAD
        self.num_chunks = ceildiv(mesh.num_v_elements, self.chunk_size)
        if self.num_chunks > 65535:
            raise Error("VlasovMoments: more than 65535 velocity chunks")
        self.partials_len = (
            mesh.num_x_elements * NXN * self.num_chunks * VLASOV_MOMENT_NC
        )
        self.partials = ctx.enqueue_create_buffer[Real.dtype](
            self.partials_len
        )

    def compute(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v],
        mut q: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, VLASOV_NC],
        mut moments: DGState[Self.XDIM, Self.P_x, VLASOV_MOMENT_NC],
    ) raises:
        """Fill `moments` (indexed by `MOM_DENSITY`, `MOM_FLUX_X/Y/Z`,
        `MOM_ENERGY`) with the velocity moments of `q`. `mesh` must be the
        mesh this object was built for."""
        comptime NXN = (Self.P_x + 1) ** Self.XDIM
        var q_layout = row_major(Int(q.buf_len))
        var node_v_layout = row_major(Int(mesh.node_v_len))
        var out_layout = row_major(Int(moments.buf_len))
        var partials_layout = row_major(Int(self.partials_len))
        var jacobian_v = Real(1.0)
        for d in range(Self.VDIM):
            jacobian_v = jacobian_v * mesh.jacobian[Self.XDIM + d]
        var chunking = PhaseInts(0)
        chunking[0] = Int64(self.chunk_size)
        chunking[1] = Int64(self.num_chunks)

        comptime kernel = _vlasov_moments_kernel[
            Self.XDIM, Self.VDIM, Self.P_x, Self.P_v,
        ]
        if self.num_chunks == 1:
            ctx.enqueue_function[kernel](
                TileTensor(q.buf, q_layout),
                TileTensor(mesh.node_v_buf, node_v_layout),
                TileTensor(moments.buf, out_layout),
                mesh.kernel_counts(), chunking, jacobian_v,
                grid_dim=(mesh.num_x_elements, 1),
                block_dim=(NXN, _moment_lanes(NXN)),
            )
            return
        ctx.enqueue_function[kernel](
            TileTensor(q.buf, q_layout),
            TileTensor(mesh.node_v_buf, node_v_layout),
            TileTensor(self.partials, partials_layout),
            mesh.kernel_counts(), chunking, jacobian_v,
            grid_dim=(mesh.num_x_elements, self.num_chunks),
            block_dim=(NXN, _moment_lanes(NXN)),
        )
        var num_x_nodes = moments.total_nodes
        ctx.enqueue_function[_sum_chunks_kernel](
            TileTensor(self.partials, partials_layout),
            TileTensor(moments.buf, out_layout),
            Int64(num_x_nodes), Int64(self.num_chunks),
            grid_dim=ceildiv(num_x_nodes, _SUM_BLOCK), block_dim=_SUM_BLOCK,
        )
