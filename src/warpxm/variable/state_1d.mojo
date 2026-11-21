# 1D multi-component DG solver state.
#
# A `DGState1D[P, NC]` owns a single device buffer holding `NC` solution
# components evaluated on the GLL nodes of a 1D mesh with polynomial degree
# `P`. Storage is laid out `[node, component]`, contiguous in components for a
# given node — this matches the access pattern of nodal DG flux kernels
# (which compute flux from all components of one node at once).
#
# Flat index:
#     buf[(e * (P + 1) + i) * NC + c] = q at element e, node i, component c
#
# Upstream WARPXM models the same idea per-component with `View1D` arrays in
# `solver_variables_kokkos.h`. We collapse those into a single contiguous
# buffer because all downstream kernels (flux assembly, RK stage combinations,
# diagnostics) prefer node-contiguous loads.

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real


struct DGState1D[P: Int, NC: Int]:
    """A 1D DG field with `NC` scalar components on each GLL node."""

    comptime num_nodes_per_elem: Int = Self.P + 1
    comptime num_components: Int = Self.NC

    var num_elements: Int
    var total_nodes: Int
    """Total number of GLL nodes across the whole mesh, i.e.
    `num_elements * (P + 1)`."""

    var buf_len: Int
    """Underlying buffer length, equal to `total_nodes * NC`."""

    var buf: DeviceBuffer[Real.dtype]
    """Flat `[node, component]` storage. See module docstring for indexing."""

    def __init__(out self, var ctx: DeviceContext, num_elements: Int) raises:
        """Allocate state for a mesh with `num_elements` 1D elements."""
        self.num_elements = num_elements
        self.total_nodes = num_elements * Self.num_nodes_per_elem
        self.buf_len = self.total_nodes * Self.num_components
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.buf_len)
        self.buf.enqueue_fill(Real(0.0))
