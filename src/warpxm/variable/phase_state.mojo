# Kinetic phase-space DG state, generic over the dimensions.
#
# `PhaseState[XDIM, VDIM, P_x, P_v, NC]` holds `NC` components per node on
# a tensor-product phase mesh with `XDIM` configuration-space axes (order
# `P_x`) followed by `VDIM` velocity axes (order `P_v`). It is the
# phase-space counterpart of `DGState[NDIM, P, NC]`; the numbering is owned
# by `PhaseMesh` (`mesh/phase_mesh.mojo`):
#
#   phase element   e = ((e_0 · n_1 + e_1) · n_2 + …) + e_last
#                   (last axis innermost; axes are x…, then v…)
#   local node      n = i_0 + s_1 · i_1 + s_2 · i_2 + …
#                   (first axis innermost; s_a is the node count of the
#                   axes before a)
#   buffer          buf[(e · nodes_per_elem + n) · NC + c]

#
# As in `DGState`, a mesh shared by several ranks appends `num_ghost` ghost
# elements after the owned ones; `num_phase_elements`, `total_nodes` and
# `buf_len` count owned data only.

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real
from warpxm.parallel.halo import HaloBuffers


struct PhaseState[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int, NC: Int]:
    """Phase-space DG field with `NC` components per phase node."""

    comptime num_x_nodes_per_elem: Int = (Self.P_x + 1) ** Self.XDIM
    comptime num_v_nodes_per_elem: Int = (Self.P_v + 1) ** Self.VDIM
    comptime num_nodes_per_elem: Int = (
        ((Self.P_x + 1) ** Self.XDIM) * ((Self.P_v + 1) ** Self.VDIM)
    )
    comptime num_components: Int = Self.NC

    var num_phase_elements: Int
    """Owned phase elements."""

    var num_ghost: Int
    """Ghost phase elements stored after the owned ones."""

    var total_nodes: Int
    """`num_phase_elements * num_nodes_per_elem`."""

    var buf_len: Int
    """`total_nodes * NC`: the owned part of `buf`."""

    var alloc_len: Int
    """Length of `buf`, ghosts included."""

    var buf: DeviceBuffer[Real.dtype]
    """Flat `[phase element, node, component]` storage."""

    var halo: Optional[HaloBuffers]
    """Wire buffers for the ghost exchange; `None` without ghosts."""

    def __init__(
        out self, var ctx: DeviceContext, num_phase_elements: Int,
        num_ghost: Int = 0,
    ) raises:
        """Allocate a zeroed state. Pass `mesh.num_phase_elements` and
        `mesh.num_ghost` (which may be left out in a driver that only ever
        runs on one rank)."""
        self.num_phase_elements = num_phase_elements
        self.num_ghost = num_ghost
        self.total_nodes = num_phase_elements * Self.num_nodes_per_elem
        self.buf_len = self.total_nodes * Self.num_components
        self.alloc_len = (
            self.buf_len
            + num_ghost * Self.num_nodes_per_elem * Self.num_components
        )
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.alloc_len)
        self.buf.enqueue_fill(Real(0.0))
        self.halo = None
        if num_ghost > 0:
            comptime face_nodes = Self.num_nodes_per_elem // (Self.P_x + 1)
            self.halo = HaloBuffers(ctx, num_ghost * face_nodes * Self.NC)

    @staticmethod
    @always_inline
    def index(elem: Int, node: Int, comp: Int) -> Int:
        """Flat buffer index of component `comp` at local node `node` of
        phase element `elem`."""
        return (elem * Self.num_nodes_per_elem + node) * Self.NC + comp
