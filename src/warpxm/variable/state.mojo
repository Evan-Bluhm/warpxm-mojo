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
#
# On a mesh shared by several ranks the buffer continues past the owned
# elements with `num_ghost` ghost elements (see `parallel/halo.mojo`).
# `num_elements`, `total_nodes` and `buf_len` count owned data only, so a
# kernel sized by them never touches a ghost; only the kernels that read a
# neighbour element (the DG core) look past `buf_len`, up to `alloc_len`.

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real
from warpxm.parallel.halo import HaloBuffers


struct DGState[NDIM: Int, P: Int, NC: Int]:
    """An `NDIM`-dimensional DG field with `NC` components per GLL node."""

    comptime num_nodes_per_elem: Int = (Self.P + 1) ** Self.NDIM
    comptime num_components: Int = Self.NC

    var num_elements: Int
    """Owned elements."""

    var num_ghost: Int
    """Ghost elements stored after the owned ones (0 on a serial mesh)."""

    var total_nodes: Int
    """`num_elements * (P+1)^NDIM`."""

    var buf_len: Int
    """`total_nodes * NC`: the owned part of `buf`."""

    var alloc_len: Int
    """Length of `buf`, ghosts included."""

    var buf: DeviceBuffer[Real.dtype]
    """Flat `[element, node, component]` storage."""

    var halo: Optional[HaloBuffers]
    """Wire buffers for the ghost exchange; `None` without ghosts."""

    def __init__(
        out self, var ctx: DeviceContext, num_elements: Int,
        num_ghost: Int = 0,
    ) raises:
        """A zeroed field. Pass the mesh's counts:
        `DGState[...](ctx, mesh.num_elements, mesh.num_ghost)`. `num_ghost`
        may be left out in a driver that only ever runs on one rank."""
        self.num_elements = num_elements
        self.num_ghost = num_ghost
        self.total_nodes = num_elements * Self.num_nodes_per_elem
        self.buf_len = self.total_nodes * Self.num_components
        self.alloc_len = (
            self.buf_len
            + num_ghost * Self.num_nodes_per_elem * Self.num_components
        )
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.alloc_len)
        self.buf.enqueue_fill(Real(0.0))
        self.halo = None
        if num_ghost > 0:
            comptime face_nodes = (Self.P + 1) ** (Self.NDIM - 1)
            self.halo = HaloBuffers(ctx, num_ghost * face_nodes * Self.NC)

    @staticmethod
    @always_inline
    def index(elem: Int, node: Int, comp: Int) -> Int:
        """Flat buffer index of component `comp` at local node `node` of
        element `elem`."""
        return (elem * Self.num_nodes_per_elem + node) * Self.NC + comp
