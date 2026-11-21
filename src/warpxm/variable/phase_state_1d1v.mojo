# 1D1V kinetic phase-space DG solver state.
#
# Mirrors upstream WARPXM's kinetic solver `ndg_kinetic_kokkos_t`, which
# operates on (physical element, velocity element) pairs. For 1D1V we have
# one physical-space dimension `x` and one velocity-space dimension `v`;
# each phase-space cell carries a tensor-product GLL grid of
# `(P_x + 1) * (P_v + 1)` nodes. `NC` is kept parametric — Vlasov-Maxwell
# itself uses `NC = 1` because the unknown is the scalar distribution
# function `f(x, v, t)`, but the buffer layout is component-generic in
# case a future kinetic app needs more.
#
# Phase elements are paired velocity-inner: `phase_e = ex * num_v + ev`.
# Within a phase element, GLL nodes follow the convention used by
# `spatial/dg_core_2d.mojo` (axis-1 outer, axis-0 inner): the velocity
# node varies slowest, the x-node varies fastest. That keeps the
# tensor-product DG kernel local-node indexing identical between the
# generic 2D core and the Vlasov 1D1V kernel.
#
# Flat index:
#     buf[(((ex * num_v + ev) * (P_v + 1) + iv) * (P_x + 1) + ix) * NC + c]
#       = f at physical element ex, velocity element ev,
#         v-node iv, x-node ix, component c
#
# Mirrors configuration-space `DGState1D[P, NC]` in `state_1d.mojo`.

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real


struct PhaseState1D1V[P_x: Int, P_v: Int, NC: Int]:
    """1D1V phase-space DG field with `NC` components per phase node."""

    comptime num_x_nodes_per_elem: Int = Self.P_x + 1
    comptime num_v_nodes_per_elem: Int = Self.P_v + 1
    comptime num_nodes_per_phase_elem: Int = (Self.P_x + 1) * (Self.P_v + 1)
    comptime num_components: Int = Self.NC

    var num_x: Int
    """Number of physical-space elements."""

    var num_v: Int
    """Number of velocity-space elements."""

    var num_phase_elements: Int
    """`num_x * num_v` — total number of (ex, ev) phase cells."""

    var total_nodes: Int
    """Total GLL nodes across the whole phase mesh, i.e.
    `num_phase_elements * (P_x + 1) * (P_v + 1)`."""

    var buf_len: Int
    """Underlying buffer length, equal to `total_nodes * NC`."""

    var buf: DeviceBuffer[Real.dtype]
    """Flat phase-space storage. See module docstring for indexing."""

    def __init__(
        out self, var ctx: DeviceContext, num_x: Int, num_v: Int,
    ) raises:
        """Allocate phase-space state for a `num_x * num_v` phase mesh."""
        self.num_x = num_x
        self.num_v = num_v
        self.num_phase_elements = num_x * num_v
        self.total_nodes = (
            self.num_phase_elements * Self.num_nodes_per_phase_elem
        )
        self.buf_len = self.total_nodes * Self.num_components
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.buf_len)
        self.buf.enqueue_fill(Real(0.0))
