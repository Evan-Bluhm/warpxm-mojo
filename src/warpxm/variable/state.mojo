# Multi-component DG solver state, generic over the spatial dimension.
#
# A `DGState[NDIM, P, NC]` owns one flat device buffer holding `NC` solution
# components on the `(P+1)^NDIM` GLL nodes of every element:
#
#     buf[(e * (P+1)^NDIM + n) * NC + c]
#
# `e` is the element index and `n` the row-major local node index, with the
# x index fastest (`n = (kn * (P+1) + jn) * (P+1) + in`). Components are
# contiguous per node because every kernel (flux assembly, RK combines,
# diagnostics) reads all components of one node at once.
#
# Nothing but the node count depends on `NDIM`, which is why the time
# steppers, the pointwise source kernels and the diagnostics are written
# once against this type. Use `index` rather than re-deriving the layout.

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real


struct DGState[NDIM: Int, P: Int, NC: Int]:
    """An `NDIM`-dimensional DG field with `NC` components per GLL node."""

    comptime num_nodes_per_elem: Int = (Self.P + 1) ** Self.NDIM
    comptime num_components: Int = Self.NC

    var num_elements: Int
    var total_nodes: Int
    """`num_elements * (P+1)^NDIM`."""

    var buf_len: Int
    """`total_nodes * NC`."""

    var buf: DeviceBuffer[Real.dtype]
    """Flat `[element, node, component]` storage."""

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.num_elements = num_elements
        self.total_nodes = num_elements * Self.num_nodes_per_elem
        self.buf_len = self.total_nodes * Self.num_components
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.buf_len)
        self.buf.enqueue_fill(Real(0.0))

    @staticmethod
    @always_inline
    def index(elem: Int, node: Int, comp: Int) -> Int:
        """Flat buffer index of component `comp` at local node `node` of
        element `elem`."""
        return (elem * Self.num_nodes_per_elem + node) * Self.NC + comp
