# 2D multi-component DG solver state.
#
# Tensor-product extension of `DGState1D[P, NC]`. Owns one flat device buffer
# of length `num_elements · (P+1)² · NC`, indexed
#
#     buf[((e * (P+1)² + n) * NC) + c]
#
# where `e` is the element index and `n = jn * (P+1) + in` is the row-major
# 2D node index inside the element (η outer, ξ inner).

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real


struct DGState2D[P: Int, NC: Int]:
    """A 2D DG field with `NC` components on (P+1)² GLL nodes per element."""

    comptime num_nodes_per_elem: Int = (Self.P + 1) * (Self.P + 1)
    comptime num_components: Int = Self.NC

    var num_elements: Int
    var total_nodes: Int
    """num_elements · (P+1)²."""

    var buf_len: Int
    """total_nodes · NC."""

    var buf: DeviceBuffer[Real.dtype]

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        self.num_elements = num_elements
        self.total_nodes = num_elements * Self.num_nodes_per_elem
        self.buf_len = self.total_nodes * Self.num_components
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.buf_len)
        self.buf.enqueue_fill(Real(0.0))
