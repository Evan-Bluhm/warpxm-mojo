# 1D2V kinetic phase-space DG solver state.
#
# Extends the 1D1V state to one physical-space dimension `x` and two
# velocity-space dimensions `v1` (≡ `v_x`) and `v2` (≡ `v_y`).
# Phase elements are paired with v2 innermost:
#
#     phase_e = (ex · num_v1 + ev1) · num_v2 + ev2.
#
# Within a phase element, GLL nodes follow the axis-2 outer / axis-1
# middle / axis-0 inner convention used by `spatial/dg_core_3d.mojo`,
# so the eventual 1D2V DG kernel can mirror the 3D pattern closely.
# Adopting it now keeps the indexing consistent across the port.
#
# Flat index:
#     buf[ ((((phase_e · (P_v2 + 1) + iv2) · (P_v1 + 1) + iv1)
#              · (P_x + 1) + ix) · NC + c ]
#       = f at physical element ex, velocity elements (ev1, ev2),
#         GLL nodes (iv2, iv1, ix), component c.
#
# Mirrors 1D1V `PhaseState1D1V[P_x, P_v, NC]` in
# `variable/phase_state_1d1v.mojo`.

from std.gpu.host import DeviceBuffer, DeviceContext
from warpxm.core.types import Real


struct PhaseState1D2V[P_x: Int, P_v1: Int, P_v2: Int, NC: Int]:
    """1D2V phase-space DG field with `NC` components per phase node."""

    comptime num_x_nodes_per_elem: Int = Self.P_x + 1
    comptime num_v1_nodes_per_elem: Int = Self.P_v1 + 1
    comptime num_v2_nodes_per_elem: Int = Self.P_v2 + 1
    comptime num_nodes_per_phase_elem: Int = (
        (Self.P_x + 1) * (Self.P_v1 + 1) * (Self.P_v2 + 1)
    )
    comptime num_components: Int = Self.NC

    var num_x: Int
    """Number of physical-space elements."""

    var num_v1: Int
    """Number of velocity elements along `v1` (= `v_x`)."""

    var num_v2: Int
    """Number of velocity elements along `v2` (= `v_y`)."""

    var num_phase_elements: Int
    """`num_x · num_v1 · num_v2` — total (ex, ev1, ev2) phase cells."""

    var total_nodes: Int
    """Total GLL nodes across the whole phase mesh."""

    var buf_len: Int
    """Underlying buffer length, equal to `total_nodes * NC`."""

    var buf: DeviceBuffer[Real.dtype]
    """Flat phase-space storage. See module docstring for indexing."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        num_x: Int, num_v1: Int, num_v2: Int,
    ) raises:
        """Allocate phase-space state for `num_x · num_v1 · num_v2`
        phase elements."""
        self.num_x = num_x
        self.num_v1 = num_v1
        self.num_v2 = num_v2
        self.num_phase_elements = num_x * num_v1 * num_v2
        self.total_nodes = (
            self.num_phase_elements * Self.num_nodes_per_phase_elem
        )
        self.buf_len = self.total_nodes * Self.num_components
        self.buf = ctx.enqueue_create_buffer[Real.dtype](self.buf_len)
        self.buf.enqueue_fill(Real(0.0))
