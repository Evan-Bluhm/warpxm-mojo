# Halo exchange between ranks.
#
# A rank's state buffer holds its owned elements followed by `num_ghost`
# ghost elements: one layer of copies of the neighbouring ranks' elements
# across each partition face. The mesh connectivity points at them like at
# any other neighbour, so the DG kernel and the limiter read a ghost
# without knowing it is one. This module keeps the ghosts current.
#
# Ghost numbering. Directions are `d = 2·axis + side` (side 0 negative,
# 1 positive), as in `elem_face_dir`. The ghosts of direction d form one
# contiguous block, directions in increasing order, and within a block
# they follow the layer of elements on that face in the mesh's own element
# order with the face's axis left out. The elements a rank sends towards d
# are listed in the same order, so the two sides of an interface agree on
# the order of the wire data without exchanging any indices. The mesh
# builds both: `BlockMesh` and `PhaseMesh` each hand their send list to
# `HaloPlan`.
#
# Wire format. Two payloads share the machinery:
#
#   trace (`TRACE = True`)   the nodes of the element on the shared face,
#                            NC values each: all the DG surface flux
#                            needs, a factor P+1 smaller than the element.
#   element (`TRACE = False`) NC values per element, for per-element data
#                            such as the limiter's cell means.
#
# The kernels are parameterised on the node layout only through `NPE`
# (nodes per element) and `P1` (nodes along a configuration axis), with
# the configuration axes innermost in the node numbering. That covers both
# `DGState` (`NPE = P1^NDIM`) and `PhaseState` (`NPE = P1^XDIM · velocity
# nodes`).
#
# Each exchange is: pack kernel, MPI, unpack kernel. With GPU-aware MPI
# the device buffers go straight to MPI; otherwise they are staged through
# pinned host buffers. A serial build compiles `exchange_halo` to nothing.

from std.collections import Array, List
from std.ffi import external_call, c_int
from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import TileTensor, row_major

from warpxm.core.tensors import RealTensor, IndexTensor
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.parallel.mpi import USE_MPI, gpu_aware


comptime _HALO_BLOCK: Int = 256


struct HaloPlan:
    """Which elements a rank exchanges with which neighbour. Owned by the
    mesh; shared by every field on it."""

    var num_owned: Int
    var num_ghost: Int

    var neighbor: Array[Int, 6]
    """Rank across each direction, -1 for none."""

    var count: Array[Int, 6]
    """Ghost elements per direction (0 where there is no neighbour)."""

    var gpu_aware: Bool
    """Hand device pointers to MPI instead of staging through the host."""

    var send_elem_buf: DeviceBuffer[INDEX_DTYPE]
    """Owned element sent in each wire slot."""

    var dir_buf: DeviceBuffer[INDEX_DTYPE]
    """Direction `2·axis + side` of each wire slot."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        num_owned: Int,
        neighbor: Array[Int, 6],
        count: Array[Int, 6],
        send_elem: List[Int32],
    ) raises:
        """`count[d]` elements are exchanged with rank `neighbor[d]` across
        direction `d` (0 where `neighbor[d] < 0`). `send_elem` lists the
        owned elements to send, direction by direction, each layer in the
        order of the matching ghost block on the other side."""
        self.num_owned = num_owned
        self.neighbor = neighbor.copy()
        self.count = count.copy()
        self.num_ghost = len(send_elem)
        self.gpu_aware = gpu_aware()

        # Device buffers cannot be empty; a serial mesh keeps one unused
        # slot.
        var slots = self.num_ghost if self.num_ghost > 0 else 1
        var elem_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](slots)
        var dir_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](slots)
        elem_host[0] = 0
        dir_host[0] = 0
        var slot = 0
        for d in range(6):
            for _ in range(count[d]):
                elem_host[slot] = send_elem[slot]
                dir_host[slot] = Int32(d)
                slot += 1
        debug_assert(slot == self.num_ghost, "send list does not match counts")
        self.send_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](slots)
        self.dir_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](slots)
        ctx.enqueue_copy(dst_buf=self.send_elem_buf, src_buf=elem_host)
        ctx.enqueue_copy(dst_buf=self.dir_buf, src_buf=dir_host)
        ctx.synchronize()

    def ghost_offset(self, d: Int) -> Int:
        """Index, among the ghosts, of the first ghost of direction `d`."""
        var off = 0
        for k in range(d):
            off += self.count[k]
        return off


struct HaloBuffers:
    """Wire buffers for one exchanged field: `len` values each way."""

    var len: Int
    var send_buf: DeviceBuffer[Real.dtype]
    var recv_buf: DeviceBuffer[Real.dtype]
    var host_send: HostBuffer[Real.dtype]
    var host_recv: HostBuffer[Real.dtype]
    """Pinned staging copies, used unless MPI is GPU-aware (and then one
    element long)."""

    def __init__(out self, var ctx: DeviceContext, len: Int) raises:
        self.len = len
        self.send_buf = ctx.enqueue_create_buffer[Real.dtype](len)
        self.recv_buf = ctx.enqueue_create_buffer[Real.dtype](len)
        var host_len = 1 if gpu_aware() else len
        self.host_send = ctx.enqueue_create_host_buffer[Real.dtype](host_len)
        self.host_recv = ctx.enqueue_create_host_buffer[Real.dtype](host_len)


def halo_values_per_elem[NPE: Int, P1: Int, NC: Int, TRACE: Bool]() -> Int:
    """Values on the wire per ghost element."""
    comptime if TRACE:
        return (NPE // P1) * NC
    else:
        return NC


@always_inline
def _face_node[P1: Int](fnode: Int, axis: Int, along: Int) -> Int:
    """Local node number of face node `fnode` on the plane `along` of
    configuration axis `axis`: the node numbering with the digit of that
    axis (stride `P1^axis`) inserted."""
    var stride = 1
    comptime for k in range(2):
        if k < axis:
            stride *= P1
    var low = fnode % stride
    return low + (along + (fnode // stride) * P1) * stride


def _halo_copy_kernel[NPE: Int, P1: Int, NC: Int, TRACE: Bool, PACK: Bool](
    field: RealTensor,
    wire: RealTensor,
    send_elem: IndexTensor,
    dirs: IndexTensor,
    num_owned_i64: Int64,
    num_slots_i64: Int64,
):
    """`PACK`: copy the face data of each sent element into its wire slot.
    Otherwise: copy each wire slot into its ghost element. One thread per
    (slot, face node)."""
    comptime ENPE = NPE if TRACE else 1
    comptime FN = NPE // P1 if TRACE else 1
    var t = Int(global_idx.x)
    if t >= Int(num_slots_i64) * FN:
        return
    var slot = t // FN
    var fnode = t % FN
    var d = Int(rebind[Scalar[INDEX_DTYPE]](dirs[slot]))

    # The sender's layer next to the face is the receiver's ghost layer:
    # towards +axis we send plane P, and the ghost on our -axis side is
    # read by the kernel at plane P as well.
    var elem: Int
    var positive: Bool
    comptime if PACK:
        elem = Int(rebind[Scalar[INDEX_DTYPE]](send_elem[slot]))
        positive = d % 2 == 1
    else:
        elem = Int(num_owned_i64) + slot
        positive = d % 2 == 0
    var node = 0
    comptime if TRACE:
        node = _face_node[P1](fnode, d // 2, P1 - 1 if positive else 0)

    var field_base = (elem * ENPE + node) * NC
    var wire_base = t * NC
    comptime for c in range(NC):
        comptime if PACK:
            wire[wire_base + c] = field[field_base + c]
        else:
            field[field_base + c] = wire[wire_base + c]


def exchange_halo[NPE: Int, P1: Int, NC: Int, TRACE: Bool](
    var ctx: DeviceContext,
    plan: HaloPlan,
    field: DeviceBuffer[Real.dtype],
    field_len: Int,
    halo: HaloBuffers,
) raises:
    """Refresh the ghost elements of `field` from the neighbouring ranks.

    `field` holds `NC` values per node (`TRACE`, `NPE` nodes per element)
    or per element, owned elements first, `field_len` values in all.
    Returns once the unpack kernel is enqueued, so later kernels see the
    new ghosts.
    """
    comptime if not USE_MPI:
        return
    if plan.num_ghost == 0:
        return
    comptime FN = NPE // P1 if TRACE else 1
    var stride = halo_values_per_elem[NPE, P1, NC, TRACE]()
    var wire_len = plan.num_ghost * stride
    debug_assert(halo.len == wire_len, "halo buffers do not match the plan")

    var field_view = TileTensor(field, row_major(Int(field_len)))
    var slot_layout = row_major(Int(plan.num_ghost))
    var send_elem = TileTensor(plan.send_elem_buf, slot_layout)
    var dirs = TileTensor(plan.dir_buf, slot_layout)
    var wire_layout = row_major(Int(wire_len))
    var grid = ceildiv(plan.num_ghost * FN, _HALO_BLOCK)

    ctx.enqueue_function[_halo_copy_kernel[NPE, P1, NC, TRACE, True]](
        field_view, TileTensor(halo.send_buf, wire_layout), send_elem, dirs,
        Int64(plan.num_owned), Int64(plan.num_ghost),
        grid_dim=grid, block_dim=_HALO_BLOCK,
    )
    if not plan.gpu_aware:
        ctx.enqueue_copy(dst_buf=halo.host_send, src_buf=halo.send_buf)
    # MPI reads the send buffer on the host thread: the pack has to be
    # done first.
    ctx.synchronize()

    var rc: c_int
    if plan.gpu_aware:
        rc = _mpi_exchange(
            plan, Int(halo.send_buf.unsafe_ptr()),
            Int(halo.recv_buf.unsafe_ptr()), stride,
        )
    else:
        rc = _mpi_exchange(
            plan, Int(halo.host_send.unsafe_ptr()),
            Int(halo.host_recv.unsafe_ptr()), stride,
        )
    if rc != 0:
        raise Error("MPI halo exchange failed, rc = " + String(Int(rc)))

    if not plan.gpu_aware:
        ctx.enqueue_copy(dst_buf=halo.recv_buf, src_buf=halo.host_recv)
    ctx.enqueue_function[_halo_copy_kernel[NPE, P1, NC, TRACE, False]](
        field_view, TileTensor(halo.recv_buf, wire_layout), send_elem, dirs,
        Int64(plan.num_owned), Int64(plan.num_ghost),
        grid_dim=grid, block_dim=_HALO_BLOCK,
    )


def _mpi_exchange(plan: HaloPlan, send: Int, recv: Int, stride: Int) -> c_int:
    """`send` and `recv` are the buffer addresses (host or, with GPU-aware
    MPI, device), passed as integers: the C side takes `double *`."""
    return external_call["wxm_mpi_halo_exchange", c_int](
        send, recv, c_int(stride),
        c_int(plan.neighbor[0]), c_int(plan.neighbor[1]),
        c_int(plan.neighbor[2]), c_int(plan.neighbor[3]),
        c_int(plan.neighbor[4]), c_int(plan.neighbor[5]),
        c_int(plan.count[0]), c_int(plan.count[1]), c_int(plan.count[2]),
        c_int(plan.count[3]), c_int(plan.count[4]), c_int(plan.count[5]),
    )
