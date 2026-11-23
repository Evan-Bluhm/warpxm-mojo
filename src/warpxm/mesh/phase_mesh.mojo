# Uniform phase-space block mesh, generic over the dimensions.
#
# `PhaseMesh[XDIM, VDIM, P_x, P_v]` is a tensor-product mesh with `XDIM`
# configuration-space axes (GLL order `P_x`) followed by `VDIM` velocity
# axes (GLL order `P_v`): 1D1V, 1D2V, 1D3V, 2D2V, 2D3V, 3D3V. Axes are
# always listed x first, then v, so axis `a < XDIM` is `x_a` and axis
# `XDIM + d` is `v_d`.
#
# Numbering (shared with `PhaseState`):
#
#   phase element   e = ((e_0 · n_1 + e_1) · n_2 + …) + e_last
#                   last axis innermost
#   local node      n = i_0 + s_1 · i_1 + s_2 · i_2 + …
#                   first axis innermost; the x-nodes of a phase element
#                   are therefore numbered exactly like the nodes of a
#                   `BlockMesh[XDIM, P_x]` element, and
#                   n = v_node · (P_x + 1)^XDIM + x_node
#
# The mesh is uniform and structured, so it stores no face tables: the
# kernels find a neighbour by adding or subtracting the element stride of
# the axis, and know a face is on the boundary from the element index.
# (A 1D3V mesh has eight faces per element; tables would cost about a
# tenth of the distribution function itself.)
#
# Boundaries. A configuration axis is periodic, or carries one BC kind per
# side: `BC_COPYOUT` (ghost = interior), `BC_VLASOV_OUTFLOW` (ghost = 0,
# so nothing flows in) or `BC_VLASOV_REFLECT` (specular wall: ghost =
# interior at the mirrored normal velocity; needs a velocity mesh that is
# symmetric about zero along that axis). Every velocity boundary is
# `BC_VLASOV_OUTFLOW`: the velocity domain truncates the real line where
# `f` is negligible.
#
# Several ranks. In an MPI build the configuration axes are cut between
# ranks exactly as the companion `BlockMesh` cuts them (both call
# `parallel/partition.mojo` with the same element counts); velocity space
# is never cut. `n` and the element counts are then this rank's, `lo` /
# `hi` stay global. A partition face gets the kind `BC_HALO`, and the far
# side of it is a ghost phase element stored after the owned ones, in the
# order of the owned elements on that face (`kernel_ghost_first`).
#
# The configuration-space fields (Maxwell state, moments) live on a
# `BlockMesh[XDIM, P_x]` with the same extents; see `x_block_element` for
# the element correspondence when XDIM > 1.

from std.collections import Array, List
from max.gpu.host import DeviceBuffer, DeviceContext

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import (
    BC_INTERIOR, BC_COPYOUT, BC_VLASOV_OUTFLOW, BC_VLASOV_REFLECT, BC_HALO,
)
from warpxm.core.math_compat import fabs, fmax
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.block import BlockMesh
from warpxm.parallel import mpi
from warpxm.parallel.halo import HaloPlan
from warpxm.parallel.partition import Partition, decompose
from warpxm.variable.phase_state import PhaseState


comptime PHASE_MAX_AXES: Int = 8
"""Lanes in the per-axis vectors handed to the kinetic kernels (six axes
at most are used)."""

comptime PhaseInts = SIMD[DType.int64, PHASE_MAX_AXES]
comptime PhaseReals = SIMD[WXM_DTYPE, PHASE_MAX_AXES]


struct PhaseMesh[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int]:
    """Uniform phase-space DG mesh with tensor-product GLL nodes."""

    comptime ND: Int = Self.XDIM + Self.VDIM
    comptime num_x_nodes_per_elem: Int = (Self.P_x + 1) ** Self.XDIM
    comptime num_v_nodes_per_elem: Int = (Self.P_v + 1) ** Self.VDIM
    comptime num_nodes_per_elem: Int = (
        ((Self.P_x + 1) ** Self.XDIM) * ((Self.P_v + 1) ** Self.VDIM)
    )

    var n: Array[Int, Self.ND]
    """Elements along each axis (x axes, then v axes), on this rank."""

    var lo: Array[Real, Self.ND]
    var hi: Array[Real, Self.ND]
    var dx: Array[Real, Self.ND]
    """Element width along each axis."""

    var jacobian: Array[Real, Self.ND]
    """Reference-to-physical Jacobian along each axis, `dx / 2`."""

    var periodic: Array[Bool, Self.XDIM]
    var bc: Array[Int, 2 * Self.XDIM]
    """`bc[2·a + side]`: BC kind on the negative (0) or positive (1) end of
    configuration axis `a` (`BC_COPYOUT`, `BC_VLASOV_OUTFLOW` or
    `BC_VLASOV_REFLECT`). Ignored on a periodic axis."""

    var num_x_elements: Int
    var num_v_elements: Int
    var num_phase_elements: Int
    """Elements owned by this rank."""

    var num_ghost: Int
    """Ghost phase elements across the partition faces."""

    var partition: Partition
    """This rank's box of the configuration mesh (as on the companion
    `BlockMesh`)."""

    var halo: HaloPlan

    var node_v_len: Int
    var node_v_buf: DeviceBuffer[Real.dtype]
    """Velocity node coordinates, one run per velocity axis laid end to
    end: `[offset_d + e_d · (P_v + 1) + i_d]`, with `offset_d` the total
    length of the runs before axis `d`."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        *,
        lo: Array[Real, Self.ND],
        hi: Array[Real, Self.ND],
        n: Array[Int, Self.ND],
        periodic: Array[Bool, Self.XDIM],
        bc: Array[Int, 2 * Self.XDIM],
    ) raises:
        """Mesh on `[lo, hi]` with `n` elements per axis; axes are the
        `XDIM` configuration axes followed by the `VDIM` velocity axes.
        `bc[2·a + side]` applies to non-periodic configuration axes."""
        comptime assert Self.XDIM >= 1 and Self.XDIM <= 3, "XDIM must be 1, 2 or 3"
        comptime assert Self.VDIM >= 1 and Self.VDIM <= 3, "VDIM must be 1, 2 or 3"
        comptime assert Self.VDIM >= Self.XDIM, (
            "need a velocity axis for every configuration axis"
        )
        comptime assert Self.num_nodes_per_elem <= 1024, (
            "one phase element must fit in one GPU block (1024 threads)"
        )
        comptime P1v = Self.P_v + 1

        # This rank's box: the configuration axes are cut like the
        # companion BlockMesh, the velocity axes not at all.
        var x_n: Array[Int, 3] = [1, 1, 1]
        var x_periodic: Array[Bool, 3] = [True, True, True]
        for a in range(Self.XDIM):
            x_n[a] = n[a]
            x_periodic[a] = periodic[a]
        self.partition = decompose(
            x_n, x_periodic, Self.XDIM, mpi.rank(), mpi.size(),
        )
        self.n = n.copy()
        for a in range(Self.XDIM):
            self.n[a] = self.partition.n[a]
        self.lo = lo.copy()
        self.hi = hi.copy()
        self.periodic = periodic.copy()
        self.bc = bc.copy()
        self.dx = Array[Real, Self.ND](fill=Real(0.0))
        self.jacobian = Array[Real, Self.ND](fill=Real(0.0))
        self.num_x_elements = 1
        self.num_v_elements = 1
        self.node_v_len = 0
        for a in range(Self.ND):
            if n[a] < 1 or not (hi[a] > lo[a]):
                raise Error("PhaseMesh: axis " + String(a) + " is empty")
            # The launch grid carries the last two axes in its y and z
            # extents, which the device caps at 65535.
            if n[a] > 65535:
                raise Error("PhaseMesh: more than 65535 elements on an axis")
            self.dx[a] = (hi[a] - lo[a]) / Real(n[a])
            self.jacobian[a] = self.dx[a] * Real(0.5)
            if a < Self.XDIM:
                self.num_x_elements *= self.n[a]
            else:
                self.num_v_elements *= n[a]
                self.node_v_len += n[a] * P1v
        self.num_phase_elements = self.num_x_elements * self.num_v_elements
        if self.num_phase_elements >= 2147483647:
            raise Error("PhaseMesh: too many phase elements for a 32-bit index")
        for s in range(2 * Self.XDIM):
            if self.periodic[s // 2]:
                continue
            if not (
                bc[s] == BC_COPYOUT or bc[s] == BC_VLASOV_OUTFLOW
                or bc[s] == BC_VLASOV_REFLECT
            ):
                raise Error(
                    "PhaseMesh: configuration BC must be BC_COPYOUT,"
                    + " BC_VLASOV_OUTFLOW or BC_VLASOV_REFLECT"
                )
            # A specular wall maps v_n to −v_n, which must land on a node.
            var va = Self.XDIM + s // 2
            if bc[s] == BC_VLASOV_REFLECT and (
                fabs(lo[va] + hi[va]) > Real(1.0e-12) * (hi[va] - lo[va])
            ):
                raise Error(
                    "PhaseMesh: BC_VLASOV_REFLECT on configuration axis "
                    + String(s // 2)
                    + " needs a velocity mesh symmetric about zero on the"
                    + " matching velocity axis"
                )

        # Phase elements exchanged across each partition face: the layer
        # of owned elements next to it, in increasing element order.
        var halo_count = Array[Int, 6](fill=0)
        var send_elem = List[Int32]()
        for d in range(2 * Self.XDIM):
            if self.partition.neighbor[d] < 0:
                continue
            var a = d // 2
            var stride = 1
            for b in range(a + 1, Self.ND):
                stride *= self.n[b]
            var layer = self.n[a] - 1 if d % 2 == 1 else 0
            halo_count[d] = self.num_phase_elements // self.n[a]
            for e in range(self.num_phase_elements):
                if (e // stride) % self.n[a] == layer:
                    send_elem.append(Int32(e))
        self.halo = HaloPlan(
            ctx, self.num_phase_elements, self.partition.neighbor,
            halo_count, send_elem,
        )
        self.num_ghost = self.halo.num_ghost

        var ref_v = NodalBasis1D[Self.P_v].nodes()
        var node_v_host = ctx.enqueue_create_host_buffer[Real.dtype](
            self.node_v_len
        )
        var offset = 0
        for d in range(Self.VDIM):
            var a = Self.XDIM + d
            for e in range(n[a]):
                var vl = lo[a] + Real(e) * self.dx[a]
                for i in range(P1v):
                    node_v_host[offset + e * P1v + i] = (
                        vl + (Real(1.0) + ref_v[i]) * self.jacobian[a]
                    )
            offset += n[a] * P1v
        self.node_v_buf = ctx.enqueue_create_buffer[Real.dtype](self.node_v_len)
        ctx.enqueue_copy(dst_buf=self.node_v_buf, src_buf=node_v_host)

    def __init__(
        out self,
        var ctx: DeviceContext,
        *,
        lo: Array[Real, Self.ND],
        hi: Array[Real, Self.ND],
        n: Array[Int, Self.ND],
        periodic: Array[Bool, Self.XDIM],
    ) raises:
        """As above with `BC_COPYOUT` on every non-periodic configuration
        boundary."""
        self = Self(
            ctx, lo=lo, hi=hi, n=n, periodic=periodic,
            bc=Array[Int, 2 * Self.XDIM](fill=BC_COPYOUT),
        )

    # -----------------------------------------------------------------
    # Geometry queries
    # -----------------------------------------------------------------

    def x_block_element(self, phase_e: Int) -> Int:
        """Element of the companion `BlockMesh[XDIM, P_x]` that phase
        element `phase_e` sits over. `BlockMesh` numbers elements with x
        innermost (`(k · ny + j) · nx + i`), the phase mesh with the last
        axis innermost, so for XDIM > 1 the two differ."""
        var xe_phase = phase_e // self.num_v_elements
        var out = 0
        var stride = 1
        var rem = xe_phase
        # Peel the phase-ordered index from its innermost (last) x axis.
        var e_axis = Array[Int, Self.XDIM](fill=0)
        for k in range(Self.XDIM):
            var a = Self.XDIM - 1 - k
            e_axis[a] = rem % self.n[a]
            rem = rem // self.n[a]
        for a in range(Self.XDIM):
            out += e_axis[a] * stride
            stride *= self.n[a]
        return out

    def check_companion(
        self, x_mesh: BlockMesh[Self.XDIM, Self.P_x],
    ) raises:
        """Raise unless `x_mesh` is the configuration-space mesh of this
        phase mesh: the same extent, element counts and periodicity on
        every configuration axis. The fields and moments of a coupled
        run live on `x_mesh` and are indexed from here by element number,
        so a mismatch would silently pair the wrong elements. Boundary
        kinds are not compared (a specular wall here goes with, say,
        `BC_PEC_WALL` there), only whether each axis is periodic."""
        var x_n = [x_mesh.nx, x_mesh.ny, x_mesh.nz]
        var x_lo = [x_mesh.x_min, x_mesh.y_min, x_mesh.z_min]
        var x_hi = [x_mesh.x_max, x_mesh.y_max, x_mesh.z_max]
        var x_periodic = [
            x_mesh.periodic_x, x_mesh.periodic_y, x_mesh.periodic_z,
        ]
        for a in range(Self.XDIM):
            var axis = "configuration axis " + String(a)
            if x_n[a] != self.n[a]:
                raise Error(
                    "PhaseMesh.check_companion: " + axis + " has "
                    + String(self.n[a]) + " elements, the BlockMesh "
                    + String(x_n[a])
                )
            var tol = Real(1.0e-12) * (self.hi[a] - self.lo[a])
            if (
                fabs(x_lo[a] - self.lo[a]) > tol
                or fabs(x_hi[a] - self.hi[a]) > tol
            ):
                raise Error(
                    "PhaseMesh.check_companion: " + axis
                    + " spans a different interval on the BlockMesh"
                )
            if x_periodic[a] != self.periodic[a]:
                raise Error(
                    "PhaseMesh.check_companion: " + axis
                    + " is periodic on one mesh and bounded on the other"
                )

    def phase_volume(self) -> Real:
        var v = Real(1.0)
        for a in range(Self.ND):
            v = v * (self.hi[a] - self.lo[a])
        return v

    def v_abs_max(self, d: Int) -> Real:
        """Largest `|v_d|` on the mesh."""
        return fmax(fabs(self.lo[Self.XDIM + d]), fabs(self.hi[Self.XDIM + d]))

    def kernel_counts(self) -> PhaseInts:
        """Elements per axis, for the kernels."""
        var out = PhaseInts(1)
        for a in range(Self.ND):
            out[a] = Int64(self.n[a])
        return out

    def kernel_bc(self) -> PhaseInts:
        """Kind of each configuration-axis side of this rank's box:
        `BC_HALO` for a face shared with another rank, `BC_INTERIOR` for a
        periodic axis that wraps onto the rank itself, else the BC."""
        var out = PhaseInts(Int64(BC_INTERIOR))
        for s in range(2 * Self.XDIM):
            if self.partition.neighbor[s] >= 0:
                out[s] = Int64(BC_HALO)
            elif not self.periodic[s // 2]:
                out[s] = Int64(self.bc[s])
        return out

    def kernel_ghost_first(self) -> PhaseInts:
        """Index of the first ghost phase element of each configuration
        side (meaningful where `kernel_bc` says `BC_HALO`)."""
        var out = PhaseInts(0)
        for s in range(2 * Self.XDIM):
            out[s] = Int64(self.num_phase_elements + self.halo.ghost_offset(s))
        return out

    def kernel_inv_jacobian(self) -> PhaseReals:
        var out = PhaseReals(0.0)
        for a in range(Self.ND):
            out[a] = Real(1.0) / self.jacobian[a]
        return out

    @inline(.never)
    def _node(self, e: Int, node: Int) -> Array[Real, 7]:
        """Coordinates of local node `node` of phase element `e` in lanes
        `0 .. ND - 1` and its quadrature weight (GLL weights times the
        element Jacobians) in lane 6."""
        var ref_x = NodalBasis1D[Self.P_x].nodes()
        var ref_v = NodalBasis1D[Self.P_v].nodes()
        var w_x = NodalBasis1D[Self.P_x].weights()
        var w_v = NodalBasis1D[Self.P_v].weights()
        var out = Array[Real, 7](fill=Real(0.0))
        var w = Real(1.0)
        # Element index: last axis innermost.
        var e_axis = Array[Int, Self.ND](fill=0)
        var rem_e = e
        for k in range(Self.ND):
            var a = Self.ND - 1 - k
            e_axis[a] = rem_e % self.n[a]
            rem_e = rem_e // self.n[a]
        # Node index: first axis innermost.
        var rem_n = node
        for a in range(Self.ND):
            var p1 = Self.P_x + 1 if a < Self.XDIM else Self.P_v + 1
            var i = rem_n % p1
            rem_n = rem_n // p1
            var xi = ref_x[i] if a < Self.XDIM else ref_v[i]
            var wi = w_x[i] if a < Self.XDIM else w_v[i]
            # Element position in the whole mesh along this axis.
            var e_global = e_axis[a]
            if a < Self.XDIM:
                e_global += self.partition.start[a]
            out[a] = (
                self.lo[a] + Real(e_global) * self.dx[a]
                + (Real(1.0) + xi) * self.jacobian[a]
            )
            w = w * wi * self.jacobian[a]
        out[6] = w
        return out^

    # -----------------------------------------------------------------
    # Host-side helpers: initial conditions and diagnostics. They own the
    # phase-node ordering and the buffer layout; component 0 (the
    # distribution function) is the one read and written. Each copies the
    # whole state to the host, so use them for setup and diagnostics, not
    # inside the time loop of a large run.
    #
    # `fill` and `visit_nodes` take a closure of the node coordinates (x
    # axes first, then v axes), so each has one overload per total number
    # of axes: 2 (1D1V), 3 (1D2V), 4 (1D3V, 2D2V), 5 (2D3V), 6 (3D3V).
    # -----------------------------------------------------------------

    def fill[NC: Int, IC: def(Real, Real) -> Real](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        ic: IC,
    ) raises:
        """Set the distribution function to `ic(x, v1)` at every node."""
        comptime assert Self.ND == 2, "a two-argument closure is for 1D1V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    h[f.index(e, n, 0)] = ic(p[0], p[1])

    def fill[NC: Int, IC: def(Real, Real, Real) -> Real](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        ic: IC,
    ) raises:
        """Set the distribution function to `ic(x, v1, v2)`."""
        comptime assert Self.ND == 3, "a three-argument closure is for 1D2V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    h[f.index(e, n, 0)] = ic(p[0], p[1], p[2])

    def fill[NC: Int, IC: def(Real, Real, Real, Real) -> Real](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        ic: IC,
    ) raises:
        """Set the distribution function to `ic(x, v1, v2, v3)` (1D3V) or
        `ic(x, y, v1, v2)` (2D2V)."""
        comptime assert Self.ND == 4, (
            "a four-argument closure is for 1D3V or 2D2V"
        )
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    h[f.index(e, n, 0)] = ic(p[0], p[1], p[2], p[3])

    def fill[NC: Int, IC: def(Real, Real, Real, Real, Real) -> Real](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        ic: IC,
    ) raises:
        """Set the distribution function to `ic(x, y, v1, v2, v3)`."""
        comptime assert Self.ND == 5, "a five-argument closure is for 2D3V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    h[f.index(e, n, 0)] = ic(p[0], p[1], p[2], p[3], p[4])

    def fill[NC: Int, IC: def(Real, Real, Real, Real, Real, Real) -> Real](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        ic: IC,
    ) raises:
        """Set the distribution function to `ic(x, y, z, v1, v2, v3)`."""
        comptime assert Self.ND == 6, "a six-argument closure is for 3D3V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    h[f.index(e, n, 0)] = ic(
                        p[0], p[1], p[2], p[3], p[4], p[5],
                    )

    def visit_nodes[NC: Int, Visit: def(Real, Real, Real, Real) -> None](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        visit: Visit,
    ) raises:
        """Call `visit(x, v1, w, f_node)` at every phase node. `w` is the
        phase-space quadrature weight."""
        comptime assert Self.ND == 2, "visit(x, v1, w, f) is for 1D1V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    visit(p[0], p[1], p[6], h[f.index(e, n, 0)])

    def visit_nodes[NC: Int, Visit: def(Real, Real, Real, Real, Real) -> None](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        visit: Visit,
    ) raises:
        """Call `visit(x, v1, v2, w, f_node)` at every phase node."""
        comptime assert Self.ND == 3, "visit(x, v1, v2, w, f) is for 1D2V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    visit(p[0], p[1], p[2], p[6], h[f.index(e, n, 0)])

    def visit_nodes[
        NC: Int, Visit: def(Real, Real, Real, Real, Real, Real) -> None,
    ](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        visit: Visit,
    ) raises:
        """Call `visit(x, v1, v2, v3, w, f_node)` (1D3V) or
        `visit(x, y, v1, v2, w, f_node)` (2D2V) at every phase node."""
        comptime assert Self.ND == 4, "this visit is for 1D3V or 2D2V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    visit(p[0], p[1], p[2], p[3], p[6], h[f.index(e, n, 0)])

    def visit_nodes[
        NC: Int, Visit: def(Real, Real, Real, Real, Real, Real, Real) -> None,
    ](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        visit: Visit,
    ) raises:
        """Call `visit(x, y, v1, v2, v3, w, f_node)` at every phase node."""
        comptime assert Self.ND == 5, "this visit is for 2D3V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    visit(
                        p[0], p[1], p[2], p[3], p[4], p[6],
                        h[f.index(e, n, 0)],
                    )

    def visit_nodes[
        NC: Int,
        Visit: def(Real, Real, Real, Real, Real, Real, Real, Real) -> None,
    ](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
        visit: Visit,
    ) raises:
        """Call `visit(x, y, z, v1, v2, v3, w, f_node)` at every phase
        node."""
        comptime assert Self.ND == 6, "this visit is for 3D3V"
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    visit(
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                        h[f.index(e, n, 0)],
                    )

    def checksum[NC: Int](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
    ) raises -> UInt64:
        """A hash of every value of `f` over the whole mesh that does not
        depend on how the mesh is split between ranks (see
        `BlockMesh.checksum`)."""
        var total: UInt64 = 0
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                # Index of this element in the whole mesh.
                var ge = 0
                var radix = 1
                var rem = e
                for k in range(Self.ND):
                    var a = Self.ND - 1 - k
                    var pos = rem % self.n[a]
                    var count = self.n[a]
                    rem = rem // self.n[a]
                    if a < Self.XDIM:
                        pos += self.partition.start[a]
                        count = self.partition.global_n[a]
                    ge += pos * radix
                    radix *= count
                for n in range(Self.num_nodes_per_elem):
                    for c in range(NC):
                        var key = UInt64(
                            (ge * Self.num_nodes_per_elem + n) * NC + c
                        )
                        var bits = rebind[UInt64](h[f.index(e, n, c)].to_bits())
                        var z = bits ^ (key * 0x9E3779B97F4A7C15)
                        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                        total += z ^ (z >> 31)
        return mpi.global_sum(total)

    def integrate[NC: Int](
        self,
        mut f: PhaseState[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, NC],
    ) raises -> Real:
        """`∫ f dx dv` over the whole phase mesh (the total mass)."""
        var acc = Real(0.0)
        with f.buf.map_to_host() as h:
            for e in range(self.num_phase_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    acc += p[6] * h[f.index(e, n, 0)]
        return mpi.global_sum(acc)
