# 2D uniform block mesh on the GPU.
#
# Tensor-product extension of `BlockMesh1D` to two dimensions. The domain is
# [x_min, x_max] × [y_min, y_max] discretized into `Nx × Ny` quad elements
# numbered row-major: element index e = j * Nx + i for (i, j) ∈ [0, Nx) × [0, Ny).
#
# Nodes per element: (P + 1)². Stored row-major in (ξ, η):
#   local node index n = jn * (P + 1) + in   for (in, jn) ∈ [0, P+1)²
#
# Faces:
#   Axis 0 (ξ-axis faces): separate two elements whose `i` coordinate differs
#                          by 1. Outward normal is in the ±x direction.
#   Axis 1 (η-axis faces): separate two elements whose `j` coordinate differs
#                          by 1. Outward normal is in the ±y direction.
#
# Face numbering convention:
#   First Nx·Ny ξ-axis faces (periodic: face index = j*Nx + i links
#   element (i-1 mod Nx, j) on the negative side to (i, j) on the positive
#   side). Then Nx·Ny η-axis faces (face index = Nx*Ny + i + j*Nx links
#   element (i, j-1 mod Ny) on the negative side to (i, j) on the positive
#   side).
#
# Per-element face table `elem_face_dir[e * 4 + dir]`:
#   dir 0 → left   face (axis 0, negative side of element)
#   dir 1 → right  face (axis 0, positive side of element)
#   dir 2 → bottom face (axis 1, negative side of element)
#   dir 3 → top    face (axis 1, positive side of element)

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT
from warpxm.core.types import Real, INDEX_DTYPE

# Negative sentinels (shared with `mesh.block_1d` for consistency).
comptime GHOST_NEG: Int32 = -1
comptime GHOST_POS: Int32 = -2

# Local face direction indices (as stored in elem_face_dir).
comptime FACE_DIR_LEFT: Int = 0
comptime FACE_DIR_RIGHT: Int = 1
comptime FACE_DIR_BOTTOM: Int = 2
comptime FACE_DIR_TOP: Int = 3


struct BlockMesh2D[P: Int]:
    """Uniform 2D DG block mesh with (P+1)² GLL nodes per quad element."""

    comptime num_nodes_per_elem: Int = (Self.P + 1) * (Self.P + 1)
    comptime num_nodes_per_face: Int = Self.P + 1

    var nx: Int
    """Number of elements along x."""

    var ny: Int
    """Number of elements along y."""

    var num_elements: Int
    """nx * ny."""

    var num_faces_axis0: Int
    """Number of ξ-axis faces."""

    var num_faces_axis1: Int
    """Number of η-axis faces."""

    var num_faces: Int
    """num_faces_axis0 + num_faces_axis1."""

    var periodic_x: Bool
    var periodic_y: Bool

    var x_min: Real
    var x_max: Real
    var y_min: Real
    var y_max: Real
    var dx: Real
    """Element width along x (= (x_max - x_min)/nx)."""

    var dy: Real
    """Element width along y."""

    var jacobian_x: Real
    """Reference-to-physical Jacobian along x, equal to dx / 2."""

    var jacobian_y: Real
    """Reference-to-physical Jacobian along y, equal to dy / 2."""

    var node_xy_buf: DeviceBuffer[Real.dtype]
    """Physical (x, y) coords of every node, length total_nodes * 2.

    Indexed `[(e * num_nodes_per_elem + n) * 2 + d]` for d ∈ {0, 1}.
    """

    var face_neg_elem_buf: DeviceBuffer[INDEX_DTYPE]
    """Element index on the face's negative side (size num_faces)."""

    var face_pos_elem_buf: DeviceBuffer[INDEX_DTYPE]
    """Element index on the face's positive side (size num_faces)."""

    var elem_face_dir_buf: DeviceBuffer[INDEX_DTYPE]
    """Per-element face indices: elem_face_dir[e * 4 + dir]."""

    var face_bc_buf: DeviceBuffer[INDEX_DTYPE]
    """BC kind per face (see core/bc.mojo). `BC_INTERIOR` on interior faces."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real,
        x_max: Real,
        y_min: Real,
        y_max: Real,
        nx: Int,
        ny: Int,
        periodic_x: Bool,
        periodic_y: Bool,
        bc_left: Int = BC_COPYOUT,
        bc_right: Int = BC_COPYOUT,
        bc_bottom: Int = BC_COPYOUT,
        bc_top: Int = BC_COPYOUT,
    ) raises:
        self.nx = nx
        self.ny = ny
        self.num_elements = nx * ny
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dx = (x_max - x_min) / Real(nx)
        self.dy = (y_max - y_min) / Real(ny)
        self.jacobian_x = self.dx * Real(0.5)
        self.jacobian_y = self.dy * Real(0.5)

        self.num_faces_axis0 = nx * ny if periodic_x else (nx + 1) * ny
        self.num_faces_axis1 = nx * ny if periodic_y else nx * (ny + 1)
        self.num_faces = self.num_faces_axis0 + self.num_faces_axis1

        # --- node coordinates ---------------------------------------------------
        var P1 = Self.P + 1
        var ref_nodes = NodalBasis1D[Self.P].nodes()
        var total_nodes = self.num_elements * Self.num_nodes_per_elem
        var node_host = ctx.enqueue_create_host_buffer[Real.dtype](total_nodes * 2)
        for j in range(ny):
            var yl = y_min + Real(j) * self.dy
            for i in range(nx):
                var xl = x_min + Real(i) * self.dx
                var e = j * nx + i
                for jn in range(P1):
                    var y = yl + (Real(1.0) + ref_nodes[jn]) * self.jacobian_y
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * self.jacobian_x
                        var n = jn * P1 + in_
                        var idx = (e * Self.num_nodes_per_elem + n) * 2
                        node_host[idx + 0] = x
                        node_host[idx + 1] = y
        self.node_xy_buf = ctx.enqueue_create_buffer[Real.dtype](total_nodes * 2)
        ctx.enqueue_copy(dst_buf=self.node_xy_buf, src_buf=node_host)

        # --- face connectivity --------------------------------------------------
        var fn_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var fp_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var efd_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_elements * 4)

        # Initialize per-elem face dir to ghost sentinels (we'll overwrite).
        for k in range(self.num_elements * 4):
            efd_host[k] = GHOST_NEG

        # Axis-0 faces: parameterized by (i_face, j) where i_face is the face's
        # x-grid index. Periodic: i_face ∈ [0, nx). Non-periodic: i_face ∈ [0, nx].
        var nx_faces_per_row = nx if periodic_x else nx + 1
        for j in range(ny):
            for i_face in range(nx_faces_per_row):
                var f_idx = j * nx_faces_per_row + i_face
                if periodic_x:
                    var neg_i = (i_face - 1 + nx) % nx
                    var pos_i = i_face
                    fn_host[f_idx] = Int32(j * nx + neg_i)
                    fp_host[f_idx] = Int32(j * nx + pos_i)
                    efd_host[(j * nx + neg_i) * 4 + FACE_DIR_RIGHT] = Int32(f_idx)
                    efd_host[(j * nx + pos_i) * 4 + FACE_DIR_LEFT] = Int32(f_idx)
                else:
                    if i_face == 0:
                        fn_host[f_idx] = GHOST_NEG
                        fp_host[f_idx] = Int32(j * nx + 0)
                        efd_host[(j * nx + 0) * 4 + FACE_DIR_LEFT] = Int32(f_idx)
                    elif i_face == nx:
                        fn_host[f_idx] = Int32(j * nx + (nx - 1))
                        fp_host[f_idx] = GHOST_POS
                        efd_host[(j * nx + (nx - 1)) * 4 + FACE_DIR_RIGHT] = Int32(f_idx)
                    else:
                        var neg_i = i_face - 1
                        var pos_i = i_face
                        fn_host[f_idx] = Int32(j * nx + neg_i)
                        fp_host[f_idx] = Int32(j * nx + pos_i)
                        efd_host[(j * nx + neg_i) * 4 + FACE_DIR_RIGHT] = Int32(f_idx)
                        efd_host[(j * nx + pos_i) * 4 + FACE_DIR_LEFT] = Int32(f_idx)

        # Axis-1 faces: parameterized by (i, j_face). Axis-1 face indices start
        # at offset self.num_faces_axis0.
        var ny_faces_per_col = ny if periodic_y else ny + 1
        for j_face in range(ny_faces_per_col):
            for i in range(nx):
                var f_idx = self.num_faces_axis0 + j_face * nx + i
                if periodic_y:
                    var neg_j = (j_face - 1 + ny) % ny
                    var pos_j = j_face
                    fn_host[f_idx] = Int32(neg_j * nx + i)
                    fp_host[f_idx] = Int32(pos_j * nx + i)
                    efd_host[(neg_j * nx + i) * 4 + FACE_DIR_TOP] = Int32(f_idx)
                    efd_host[(pos_j * nx + i) * 4 + FACE_DIR_BOTTOM] = Int32(f_idx)
                else:
                    if j_face == 0:
                        fn_host[f_idx] = GHOST_NEG
                        fp_host[f_idx] = Int32(0 * nx + i)
                        efd_host[(0 * nx + i) * 4 + FACE_DIR_BOTTOM] = Int32(f_idx)
                    elif j_face == ny:
                        fn_host[f_idx] = Int32((ny - 1) * nx + i)
                        fp_host[f_idx] = GHOST_POS
                        efd_host[((ny - 1) * nx + i) * 4 + FACE_DIR_TOP] = Int32(f_idx)
                    else:
                        var neg_j = j_face - 1
                        var pos_j = j_face
                        fn_host[f_idx] = Int32(neg_j * nx + i)
                        fp_host[f_idx] = Int32(pos_j * nx + i)
                        efd_host[(neg_j * nx + i) * 4 + FACE_DIR_TOP] = Int32(f_idx)
                        efd_host[(pos_j * nx + i) * 4 + FACE_DIR_BOTTOM] = Int32(f_idx)

        self.face_neg_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.face_pos_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.elem_face_dir_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_elements * 4)
        ctx.enqueue_copy(dst_buf=self.face_neg_elem_buf, src_buf=fn_host)
        ctx.enqueue_copy(dst_buf=self.face_pos_elem_buf, src_buf=fp_host)
        ctx.enqueue_copy(dst_buf=self.elem_face_dir_buf, src_buf=efd_host)

        # --- BC kind per face ---------------------------------------------------
        var fbc_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        for k in range(self.num_faces):
            fbc_host[k] = Int32(BC_INTERIOR)
        if not periodic_x:
            # Axis-0 boundary faces sit at i_face = 0 (left) and i_face = nx (right)
            # within each row j.
            for j in range(ny):
                fbc_host[j * nx_faces_per_row + 0] = Int32(bc_left)
                fbc_host[j * nx_faces_per_row + nx] = Int32(bc_right)
        if not periodic_y:
            # Axis-1 boundary faces sit at j_face = 0 (bottom) and j_face = ny (top)
            # within the axis-1 sub-range.
            for i in range(nx):
                fbc_host[self.num_faces_axis0 + 0 * nx + i] = Int32(bc_bottom)
                fbc_host[self.num_faces_axis0 + ny * nx + i] = Int32(bc_top)
        self.face_bc_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        ctx.enqueue_copy(dst_buf=self.face_bc_buf, src_buf=fbc_host)
