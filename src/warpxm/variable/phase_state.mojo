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

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real


struct PhaseState[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int, NC: Int]:
    """Phase-space DG field with `NC` components per phase node."""

    comptime num_x_nodes_per_elem: Int = (Self.P_x + 1) ** Self.XDIM
    comptime num_v_nodes_per_elem: Int = (Self.P_v + 1) ** Self.VDIM
    comptime num_nodes_per_elem: Int = (
        ((Self.P_x + 1) ** Self.XDIM) * ((Self.P_v + 1) ** Self.VDIM)
    )
    comptime num_components: Int = Self.NC

    var num_phase_elements: Int
    var total_nodes: Int
    """`num_phase_elements * num_nodes_per_elem`."""

    var buf_len: Int
    """`total_nodes * NC`."""

    var buf: DeviceBuffer[Real.dtype]
    """Flat `[phase element, node, component]` storage."""

    def __init__(
        out self, var ctx: DeviceContext, num_phase_elements: Int,
    ) raises:
        """Allocate a zeroed state. Pass `mesh.num_phase_elements`."""
        self.num_phase_elements = num_phase_elements
        self.total_nodes = num_phase_elements * Self.num_nodes_per_elem
        self.buf_len = self.total_nodes * Self.num_components
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.buf_len)
        self.buf.enqueue_fill(Real(0.0))

    @staticmethod
    @always_inline
    def index(elem: Int, node: Int, comp: Int) -> Int:
        """Flat buffer index of component `comp` at local node `node` of
        phase element `elem`."""
        return (elem * Self.num_nodes_per_elem + node) * Self.NC + comp
