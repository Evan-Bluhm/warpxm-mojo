# Uniform block mesh on the GPU, generic over the dimension.
#
# `BlockMesh[NDIM, P]` is a tensor-product mesh of `nx` (× `ny` (× `nz`))
# elements with `(P + 1)^NDIM` GLL nodes each. This is the analogue of
# `warpy.mesh.block(Bounds=..., NumCells=...)` upstream.
#
# Numbering (axes beyond NDIM have one element and drop out):
#
#   element   e = (k · ny + j) · nx + i
#   node      n = (kn · (P+1) + jn) · (P+1) + in      within an element
#   face      faces are grouped by axis: all axis-0 (x-normal) faces first,
#             then axis 1, then axis 2. Within an axis the index is the
#             element formula with that axis's extent replaced by its face
#             count (n for a periodic axis, n + 1 otherwise).
#
# Each face has a negative-side and a positive-side element
# (`face_neg_elem`, `face_pos_elem`). On a non-periodic boundary the
# missing side holds the sentinel `GHOST_NEG` / `GHOST_POS` and `face_bc`
# holds the BC kind for that side of the domain (see `core/bc.mojo`); the
# DG kernel synthesises the ghost state from it.
#
# `elem_face_dir[e · 2·NDIM + dir]` is the face on side `dir` of element
# `e`: dir = 2·axis for the negative side, 2·axis + 1 for the positive
# side (left, right, bottom, top, down, up).
#
# The mesh also owns the host-side helpers (`fill`, `visit_nodes`,
# `integrate`, `integrate_fn`, `l2_error`), which hide the node ordering
# and the `DGState` buffer layout from drivers.

from std.collections import Array
from max.gpu.host import DeviceBuffer, DeviceContext

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, INDEX_DTYPE
from warpxm.variable.state import DGState

# Sentinels in `face_neg_elem` / `face_pos_elem` for a face on the domain
# boundary.
comptime GHOST_NEG: Int32 = -1
comptime GHOST_POS: Int32 = -2

# Side of an element, as indexed in `elem_face_dir`.
comptime FACE_DIR_LEFT: Int = 0
comptime FACE_DIR_RIGHT: Int = 1
comptime FACE_DIR_BOTTOM: Int = 2
comptime FACE_DIR_TOP: Int = 3
comptime FACE_DIR_DOWN: Int = 4
comptime FACE_DIR_UP: Int = 5


struct BlockMesh[NDIM: Int, P: Int]:
    """Uniform DG block mesh with `(P + 1)^NDIM` GLL nodes per element."""

    comptime num_nodes_per_elem: Int = (Self.P + 1) ** Self.NDIM
    comptime num_faces_per_elem: Int = 2 * Self.NDIM

    var nx: Int
    var ny: Int
    var nz: Int
    """Elements along each axis (1 for axes beyond NDIM)."""

    var num_elements: Int
    var num_faces: Int

    var periodic_x: Bool
    var periodic_y: Bool
    var periodic_z: Bool

    var x_min: Real
    var x_max: Real
    var y_min: Real
    var y_max: Real
    var z_min: Real
    var z_max: Real

    var dx: Real
    var dy: Real
    var dz: Real
    """Element widths."""

    var jacobian_x: Real
    var jacobian_y: Real
    var jacobian_z: Real
    """Reference-to-physical Jacobians, `dx / 2` etc."""

    var node_coords_buf: DeviceBuffer[Real.dtype]
    """Node coordinates, `[(e · nodes_per_elem + n) · NDIM + d]`."""

    var face_neg_elem_buf: DeviceBuffer[INDEX_DTYPE]
    var face_pos_elem_buf: DeviceBuffer[INDEX_DTYPE]
    var elem_face_dir_buf: DeviceBuffer[INDEX_DTYPE]
    var face_bc_buf: DeviceBuffer[INDEX_DTYPE]

    # -----------------------------------------------------------------
    # Constructors: one per dimension, same argument order as upstream's
    # block generator (bounds, cell counts, periodicity, BC kinds).
    # -----------------------------------------------------------------

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real, x_max: Real,
        num_elements: Int,
        periodic: Bool,
        bc_left: Int = BC_COPYOUT, bc_right: Int = BC_COPYOUT,
    ) raises:
        """1D mesh on `[x_min, x_max]`. BC kinds apply only to a
        non-periodic axis."""
        comptime assert Self.NDIM == 1, "this is the BlockMesh[1, P] constructor"
        self = Self(
            ctx,
            lo=[x_min, Real(0.0), Real(0.0)],
            hi=[x_max, Real(1.0), Real(1.0)],
            n=[num_elements, 1, 1],
            periodic=[periodic, True, True],
            bc=[bc_left, bc_right, BC_COPYOUT, BC_COPYOUT, BC_COPYOUT, BC_COPYOUT],
        )

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real, x_max: Real,
        y_min: Real, y_max: Real,
        nx: Int, ny: Int,
        periodic_x: Bool, periodic_y: Bool,
        bc_left: Int = BC_COPYOUT, bc_right: Int = BC_COPYOUT,
        bc_bottom: Int = BC_COPYOUT, bc_top: Int = BC_COPYOUT,
    ) raises:
        """2D mesh on `[x_min, x_max] × [y_min, y_max]`."""
        comptime assert Self.NDIM == 2, "this is the BlockMesh[2, P] constructor"
        self = Self(
            ctx,
            lo=[x_min, y_min, Real(0.0)],
            hi=[x_max, y_max, Real(1.0)],
            n=[nx, ny, 1],
            periodic=[periodic_x, periodic_y, True],
            bc=[bc_left, bc_right, bc_bottom, bc_top, BC_COPYOUT, BC_COPYOUT],
        )

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real, x_max: Real,
        y_min: Real, y_max: Real,
        z_min: Real, z_max: Real,
        nx: Int, ny: Int, nz: Int,
        periodic_x: Bool = True, periodic_y: Bool = True,
        periodic_z: Bool = True,
        bc_left: Int = BC_COPYOUT, bc_right: Int = BC_COPYOUT,
        bc_bottom: Int = BC_COPYOUT, bc_top: Int = BC_COPYOUT,
        bc_down: Int = BC_COPYOUT, bc_up: Int = BC_COPYOUT,
    ) raises:
        """3D mesh on `[x_min, x_max] × [y_min, y_max] × [z_min, z_max]`."""
        comptime assert Self.NDIM == 3, "this is the BlockMesh[3, P] constructor"
        self = Self(
            ctx,
            lo=[x_min, y_min, z_min],
            hi=[x_max, y_max, z_max],
            n=[nx, ny, nz],
            periodic=[periodic_x, periodic_y, periodic_z],
            bc=[bc_left, bc_right, bc_bottom, bc_top, bc_down, bc_up],
        )

    def __init__(
        out self,
        var ctx: DeviceContext,
        *,
        lo: Array[Real, 3],
        hi: Array[Real, 3],
        n: Array[Int, 3],
        periodic: Array[Bool, 3],
        bc: Array[Int, 6],
    ) raises:
        """Dimension-generic constructor behind the three above. Axes beyond
        NDIM must have one element. `bc[2·axis + side]` is the BC kind on
        the negative (side 0) or positive (side 1) end of `axis`."""
        comptime assert Self.NDIM >= 1 and Self.NDIM <= 3, "NDIM must be 1, 2 or 3"
        comptime NPE = Self.num_nodes_per_elem
        comptime NFE = Self.num_faces_per_elem

        self.nx = n[0]
        self.ny = n[1]
        self.nz = n[2]
        self.num_elements = n[0] * n[1] * n[2]
        self.periodic_x = periodic[0]
        self.periodic_y = periodic[1]
        self.periodic_z = periodic[2]
        self.x_min = lo[0]
        self.x_max = hi[0]
        self.y_min = lo[1]
        self.y_max = hi[1]
        self.z_min = lo[2]
        self.z_max = hi[2]
        self.dx = (hi[0] - lo[0]) / Real(n[0])
        self.dy = (hi[1] - lo[1]) / Real(n[1])
        self.dz = (hi[2] - lo[2]) / Real(n[2])
        self.jacobian_x = self.dx * Real(0.5)
        self.jacobian_y = self.dy * Real(0.5)
        self.jacobian_z = self.dz * Real(0.5)

        # Faces per axis, and where each axis's faces start.
        var n_faces_along = Array[Int, 3](fill=1)
        var face_offset = Array[Int, 3](fill=0)
        var num_faces = 0
        for a in range(Self.NDIM):
            n_faces_along[a] = n[a] if periodic[a] else n[a] + 1
            face_offset[a] = num_faces
            num_faces += (self.num_elements // n[a]) * n_faces_along[a]
        self.num_faces = num_faces

        # --- node coordinates ------------------------------------------------
        comptime P1 = Self.P + 1
        var ref_nodes = NodalBasis1D[Self.P].nodes()
        var jac: Array[Real, 3] = [self.jacobian_x, self.jacobian_y, self.jacobian_z]
        var width: Array[Real, 3] = [self.dx, self.dy, self.dz]
        var total = self.num_elements * NPE * Self.NDIM
        var node_host = ctx.enqueue_create_host_buffer[Real.dtype](total)
        for e in range(self.num_elements):
            var cell: Array[Int, 3] = [
                e % n[0], (e // n[0]) % n[1], e // (n[0] * n[1]),
            ]
            for node in range(NPE):
                var stride = 1
                for a in range(Self.NDIM):
                    var corner = lo[a] + Real(cell[a]) * width[a]
                    var xi = ref_nodes[(node // stride) % P1]
                    node_host[(e * NPE + node) * Self.NDIM + a] = (
                        corner + (Real(1.0) + xi) * jac[a]
                    )
                    stride *= P1
        self.node_coords_buf = ctx.enqueue_create_buffer[Real.dtype](total)
        ctx.enqueue_copy(dst_buf=self.node_coords_buf, src_buf=node_host)

        # --- face connectivity -----------------------------------------------
        var fn_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](num_faces)
        var fp_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](num_faces)
        var fbc_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](num_faces)
        var efd_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](
            self.num_elements * NFE
        )
        for f in range(num_faces):
            fbc_host[f] = Int32(BC_INTERIOR)

        for a in range(Self.NDIM):
            # Extents of the face lattice for this axis.
            var m = n.copy()
            m[a] = n_faces_along[a]
            for k in range(m[2]):
                for j in range(m[1]):
                    for i in range(m[0]):
                        var f = face_offset[a] + (k * m[1] + j) * m[0] + i
                        # The face sits between cells (c - 1) and c along
                        # `a`, at the same position along the other axes.
                        var pos: Array[Int, 3] = [i, j, k]
                        var neg: Array[Int, 3] = [i, j, k]
                        var c = pos[a]
                        var has_neg = True
                        var has_pos = True
                        if periodic[a]:
                            neg[a] = (c - 1 + n[a]) % n[a]
                        else:
                            neg[a] = c - 1
                            has_neg = c > 0
                            has_pos = c < n[a]
                        if has_neg:
                            var neg_e = (neg[2] * n[1] + neg[1]) * n[0] + neg[0]
                            fn_host[f] = Int32(neg_e)
                            efd_host[neg_e * NFE + 2 * a + 1] = Int32(f)
                        else:
                            fn_host[f] = GHOST_NEG
                            fbc_host[f] = Int32(bc[2 * a])
                        if has_pos:
                            var pos_e = (pos[2] * n[1] + pos[1]) * n[0] + pos[0]
                            fp_host[f] = Int32(pos_e)
                            efd_host[pos_e * NFE + 2 * a] = Int32(f)
                        else:
                            fp_host[f] = GHOST_POS
                            fbc_host[f] = Int32(bc[2 * a + 1])

        self.face_neg_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](num_faces)
        self.face_pos_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](num_faces)
        self.face_bc_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](num_faces)
        self.elem_face_dir_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](
            self.num_elements * NFE
        )
        ctx.enqueue_copy(dst_buf=self.face_neg_elem_buf, src_buf=fn_host)
        ctx.enqueue_copy(dst_buf=self.face_pos_elem_buf, src_buf=fp_host)
        ctx.enqueue_copy(dst_buf=self.face_bc_buf, src_buf=fbc_host)
        ctx.enqueue_copy(dst_buf=self.elem_face_dir_buf, src_buf=efd_host)

    # -----------------------------------------------------------------
    # Geometry.
    # -----------------------------------------------------------------

    def volume(self) -> Real:
        """Measure of the domain."""
        var v = self.x_max - self.x_min
        comptime if Self.NDIM >= 2:
            v = v * (self.y_max - self.y_min)
        comptime if Self.NDIM >= 3:
            v = v * (self.z_max - self.z_min)
        return v

    def inv_dx_sum(self) -> Real:
        """`Σ_d 1 / Δx_d`, the mesh factor in the explicit CFL bound."""
        var s = Real(1.0) / self.dx
        comptime if Self.NDIM >= 2:
            s = s + Real(1.0) / self.dy
        comptime if Self.NDIM >= 3:
            s = s + Real(1.0) / self.dz
        return s

    def min_dx(self) -> Real:
        """Smallest element width."""
        var d = self.dx
        comptime if Self.NDIM >= 2:
            if self.dy < d: d = self.dy
        comptime if Self.NDIM >= 3:
            if self.dz < d: d = self.dz
        return d

    @inline(.never)
    def _node(self, e: Int, node: Int) -> Array[Real, 4]:
        """`[x, y, z, w]` of node `node` of element `e`: its coordinates
        (0 beyond NDIM) and its quadrature weight (GLL weights times the
        element Jacobian)."""
        comptime P1 = Self.P + 1
        var ref_nodes = NodalBasis1D[Self.P].nodes()
        var weights = NodalBasis1D[Self.P].weights()
        var out = Array[Real, 4](fill=Real(0.0))
        var in_ = node % P1
        out[0] = (
            self.x_min + Real(e % self.nx) * self.dx
            + (Real(1.0) + ref_nodes[in_]) * self.jacobian_x
        )
        var w = weights[in_]
        var w_jac = self.jacobian_x
        comptime if Self.NDIM >= 2:
            var jn = (node // P1) % P1
            out[1] = (
                self.y_min + Real((e // self.nx) % self.ny) * self.dy
                + (Real(1.0) + ref_nodes[jn]) * self.jacobian_y
            )
            w = w * weights[jn]
            w_jac = w_jac * self.jacobian_y
        comptime if Self.NDIM >= 3:
            var kn = node // (P1 * P1)
            out[2] = (
                self.z_min + Real(e // (self.nx * self.ny)) * self.dz
                + (Real(1.0) + ref_nodes[kn]) * self.jacobian_z
            )
            w = w * weights[kn]
            w_jac = w_jac * self.jacobian_z
        out[3] = w * w_jac
        return out^

    # -----------------------------------------------------------------
    # Host-side helpers: initial conditions and diagnostics.
    #
    # These own the node ordering and the `DGState` buffer layout, so a
    # driver or test never has to index the flat buffer itself. They copy
    # the whole state to the host, so use them for setup and diagnostics,
    # not inside the time loop of a large run.
    #
    # `fill`, `visit_nodes` and `l2_error` take a closure of the node
    # coordinates, so each has one overload per dimension: `(x)`, `(x, y)`
    # or `(x, y, z)`.
    # -----------------------------------------------------------------

    def fill[NC: Int, IC: def(Real) -> Array[Real, NC]](
        self, mut q: DGState[Self.NDIM, Self.P, NC], ic: IC,
    ) raises:
        """Set every node of `q` to `ic(x)`."""
        comptime assert Self.NDIM == 1, "ic(x) is for a 1D mesh"
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var v = ic(p[0])
                    var base = q.index(e, n, 0)
                    for c in range(NC):
                        h[base + c] = v[c]

    def fill[NC: Int, IC: def(Real, Real) -> Array[Real, NC]](
        self, mut q: DGState[Self.NDIM, Self.P, NC], ic: IC,
    ) raises:
        """Set every node of `q` to `ic(x, y)`."""
        comptime assert Self.NDIM == 2, "ic(x, y) is for a 2D mesh"
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var v = ic(p[0], p[1])
                    var base = q.index(e, n, 0)
                    for c in range(NC):
                        h[base + c] = v[c]

    def fill[NC: Int, IC: def(Real, Real, Real) -> Array[Real, NC]](
        self, mut q: DGState[Self.NDIM, Self.P, NC], ic: IC,
    ) raises:
        """Set every node of `q` to `ic(x, y, z)`."""
        comptime assert Self.NDIM == 3, "ic(x, y, z) is for a 3D mesh"
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var v = ic(p[0], p[1], p[2])
                    var base = q.index(e, n, 0)
                    for c in range(NC):
                        h[base + c] = v[c]

    def visit_nodes[
        NC: Int, Visit: def(Real, Real, Array[Real, NC]) -> None,
    ](self, mut q: DGState[Self.NDIM, Self.P, NC], visit: Visit) raises:
        """Call `visit(x, w, q_node)` at every node of `q`.

        `w` is the node's quadrature weight (GLL weight times the element
        Jacobian), so summing `w * f` over the visits gives `∫ f dV`. Use a
        closure with `mut` captures to accumulate norms, extrema or
        samples.
        """
        comptime assert Self.NDIM == 1, "visit(x, w, q) is for a 1D mesh"
        var v = Array[Real, NC](fill=Real(0.0))
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var base = q.index(e, n, 0)
                    for c in range(NC):
                        v[c] = h[base + c]
                    visit(p[0], p[3], v)

    def visit_nodes[
        NC: Int, Visit: def(Real, Real, Real, Array[Real, NC]) -> None,
    ](self, mut q: DGState[Self.NDIM, Self.P, NC], visit: Visit) raises:
        """Call `visit(x, y, w, q_node)` at every node of `q`."""
        comptime assert Self.NDIM == 2, "visit(x, y, w, q) is for a 2D mesh"
        var v = Array[Real, NC](fill=Real(0.0))
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var base = q.index(e, n, 0)
                    for c in range(NC):
                        v[c] = h[base + c]
                    visit(p[0], p[1], p[3], v)

    def visit_nodes[
        NC: Int, Visit: def(Real, Real, Real, Real, Array[Real, NC]) -> None,
    ](self, mut q: DGState[Self.NDIM, Self.P, NC], visit: Visit) raises:
        """Call `visit(x, y, z, w, q_node)` at every node of `q`."""
        comptime assert Self.NDIM == 3, "visit(x, y, z, w, q) is for a 3D mesh"
        var v = Array[Real, NC](fill=Real(0.0))
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var base = q.index(e, n, 0)
                    for c in range(NC):
                        v[c] = h[base + c]
                    visit(p[0], p[1], p[2], p[3], v)

    def integrate[NC: Int](
        self, mut q: DGState[Self.NDIM, Self.P, NC], comp: Int,
    ) raises -> Real:
        """`∫ q[comp] dV` by GLL quadrature."""
        var total: Real = 0.0
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    total += self._node(e, n)[3] * h[q.index(e, n, comp)]
        return total

    def integrate_fn[NC: Int, F: def(Array[Real, NC]) -> Real](
        self, mut q: DGState[Self.NDIM, Self.P, NC], f: F,
    ) raises -> Real:
        """`∫ f(q) dV` by GLL quadrature, for any pointwise function of the
        node state (a squared component, a pressure, an energy)."""
        var total: Real = 0.0
        var v = Array[Real, NC](fill=Real(0.0))
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var base = q.index(e, n, 0)
                    for c in range(NC):
                        v[c] = h[base + c]
                    total += self._node(e, n)[3] * f(v)
        return total

    def l2_error[NC: Int, Exact: def(Real) -> Real](
        self, mut q: DGState[Self.NDIM, Self.P, NC], comp: Int, exact: Exact,
    ) raises -> Real:
        """`sqrt(∫ (q[comp] - exact(x))² dV)`."""
        comptime assert Self.NDIM == 1, "exact(x) is for a 1D mesh"
        var total: Real = 0.0
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var d = h[q.index(e, n, comp)] - exact(p[0])
                    total += p[3] * d * d
        return sqrt(total)

    def l2_error[NC: Int, Exact: def(Real, Real) -> Real](
        self, mut q: DGState[Self.NDIM, Self.P, NC], comp: Int, exact: Exact,
    ) raises -> Real:
        """`sqrt(∫ (q[comp] - exact(x, y))² dV)`."""
        comptime assert Self.NDIM == 2, "exact(x, y) is for a 2D mesh"
        var total: Real = 0.0
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var d = h[q.index(e, n, comp)] - exact(p[0], p[1])
                    total += p[3] * d * d
        return sqrt(total)

    def l2_error[NC: Int, Exact: def(Real, Real, Real) -> Real](
        self, mut q: DGState[Self.NDIM, Self.P, NC], comp: Int, exact: Exact,
    ) raises -> Real:
        """`sqrt(∫ (q[comp] - exact(x, y, z))² dV)`."""
        comptime assert Self.NDIM == 3, "exact(x, y, z) is for a 3D mesh"
        var total: Real = 0.0
        with q.buf.map_to_host() as h:
            for e in range(self.num_elements):
                for n in range(Self.num_nodes_per_elem):
                    var p = self._node(e, n)
                    var d = h[q.index(e, n, comp)] - exact(p[0], p[1], p[2])
                    total += p[3] * d * d
        return sqrt(total)
