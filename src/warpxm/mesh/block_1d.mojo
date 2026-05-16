# 1D uniform block mesh on the GPU.
#
# This is the analogue of `warpy.mesh.block(Bounds=[xmin, xmax],
# NumCells=[N], …)` from upstream, restricted to one dimension. The mesh owns
# device buffers for node coordinates and face connectivity. We keep it as
# simple as possible: faces are numbered 0 .. NumFaces-1 with face f having
# `left_elem` on its negative-ξ side and `right_elem` on its positive-ξ side.
#
# For periodic boundaries, `NumFaces = num_elements` and faces wrap around so
# that face 0 connects element (num_elements - 1) on the left with element 0
# on the right.
#
# For non-periodic boundaries, `NumFaces = num_elements + 1` and the leftmost
# / rightmost faces have ghost-element indices encoded as the special value
# `GHOST_LEFT` / `GHOST_RIGHT` (negative sentinels). Boundary-condition
# adjusters look at these sentinels to fill in face states.

from std.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT
from warpxm.core.types import Real, INDEX_DTYPE

# Negative sentinels used in face → element tables to mark a face that touches
# the domain boundary on that side. Each side has its own sentinel so a BC
# adjuster can distinguish "left wall" from "right wall."
comptime GHOST_LEFT: Int32 = -1
comptime GHOST_RIGHT: Int32 = -2


struct BlockMesh1D[P: Int]:
    """A uniform 1D DG block mesh with P+1 GLL nodes per element."""

    comptime num_nodes_per_elem: Int = Self.P + 1

    var num_elements: Int
    var num_faces: Int
    var periodic: Bool
    var x_min: Real
    var x_max: Real
    var dx: Real
    """Per-element width (x_max - x_min) / num_elements."""

    var jacobian: Real
    """Reference-to-physical Jacobian, equal to dx / 2 for a uniform mesh."""

    # Device buffers. We hold them on the struct so they outlive any kernel
    # launch; their TileTensor views are built on demand.
    var node_x_buf: DeviceBuffer[Real.dtype]
    """Physical x-coordinate of every node, length num_elements * (P + 1)."""

    var face_left_buf: DeviceBuffer[INDEX_DTYPE]
    """face_left[f] = element on the left side of face f (or GHOST_LEFT)."""

    var face_right_buf: DeviceBuffer[INDEX_DTYPE]
    """face_right[f] = element on the right side of face f (or GHOST_RIGHT)."""

    var elem_left_face_buf: DeviceBuffer[INDEX_DTYPE]
    """elem_left_face[e] = global face index on the left side of element e."""

    var elem_right_face_buf: DeviceBuffer[INDEX_DTYPE]
    """elem_right_face[e] = global face index on the right side of element e."""

    var face_bc_buf: DeviceBuffer[INDEX_DTYPE]
    """face_bc[f] = BC kind for face f (see core/bc.mojo).

    For interior faces this is `BC_INTERIOR`. For boundary faces it's one of
    `BC_COPYOUT`, `BC_WALL_FREESLIP`, `BC_DIRICHLET`, etc. The spatial-solver
    kernel dispatches on this value to synthesize ghost states when a face
    has a GHOST_LEFT / GHOST_RIGHT sentinel on one side.
    """

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real,
        x_max: Real,
        num_elements: Int,
        periodic: Bool,
        bc_left: Int = BC_COPYOUT,
        bc_right: Int = BC_COPYOUT,
    ) raises:
        """Build the mesh, populating host arrays and uploading to device.

        `bc_left` / `bc_right` are only consulted when `periodic` is False; they
        select the BC adjuster applied at the left- and right-most boundary
        faces.
        """

        self.num_elements = num_elements
        self.periodic = periodic
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / Real(num_elements)
        self.jacobian = self.dx * Real(0.5)
        self.num_faces = num_elements if periodic else num_elements + 1

        # --- node coordinates ---------------------------------------------------
        var nodes_per_elem = Self.num_nodes_per_elem
        var ref_nodes = NodalBasis1D[Self.P].nodes()
        var total_nodes = num_elements * nodes_per_elem

        var node_x_host = ctx.enqueue_create_host_buffer[Real.dtype](total_nodes)
        for e in range(num_elements):
            var xl = x_min + Real(e) * self.dx
            for i in range(nodes_per_elem):
                # ξ ∈ [-1, 1] → physical x in [xl, xl + dx]
                var xi = ref_nodes[i]
                node_x_host[e * nodes_per_elem + i] = xl + (Real(1.0) + xi) * self.jacobian

        self.node_x_buf = ctx.enqueue_create_buffer[Real.dtype](total_nodes)
        ctx.enqueue_copy(dst_buf=self.node_x_buf, src_buf=node_x_host)

        # --- face connectivity --------------------------------------------------
        var face_left_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var face_right_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var elem_left_face_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](num_elements)
        var elem_right_face_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](num_elements)

        if periodic:
            # Face f sits at the boundary between element (f - 1 mod N) and f.
            for f in range(self.num_faces):
                var left_e = (f - 1 + num_elements) % num_elements
                var right_e = f
                face_left_host[f] = Int32(left_e)
                face_right_host[f] = Int32(right_e)
            for e in range(num_elements):
                # Element e's left face is e itself; right face is (e + 1) mod N.
                elem_left_face_host[e] = Int32(e)
                elem_right_face_host[e] = Int32((e + 1) % num_elements)
        else:
            # Face 0 is the leftmost boundary; face num_elements is the
            # rightmost. Interior face f sits between element (f - 1) and f.
            for f in range(self.num_faces):
                if f == 0:
                    face_left_host[f] = GHOST_LEFT
                    face_right_host[f] = Int32(0)
                elif f == num_elements:
                    face_left_host[f] = Int32(num_elements - 1)
                    face_right_host[f] = GHOST_RIGHT
                else:
                    face_left_host[f] = Int32(f - 1)
                    face_right_host[f] = Int32(f)
            for e in range(num_elements):
                elem_left_face_host[e] = Int32(e)
                elem_right_face_host[e] = Int32(e + 1)

        self.face_left_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.face_right_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.elem_left_face_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](num_elements)
        self.elem_right_face_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](num_elements)
        ctx.enqueue_copy(dst_buf=self.face_left_buf, src_buf=face_left_host)
        ctx.enqueue_copy(dst_buf=self.face_right_buf, src_buf=face_right_host)
        ctx.enqueue_copy(dst_buf=self.elem_left_face_buf, src_buf=elem_left_face_host)
        ctx.enqueue_copy(dst_buf=self.elem_right_face_buf, src_buf=elem_right_face_host)

        # --- BC kind per face ---------------------------------------------------
        var face_bc_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        for f in range(self.num_faces):
            face_bc_host[f] = Int32(BC_INTERIOR)
        if not periodic:
            face_bc_host[0] = Int32(bc_left)
            face_bc_host[num_elements] = Int32(bc_right)
        self.face_bc_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        ctx.enqueue_copy(dst_buf=self.face_bc_buf, src_buf=face_bc_host)
