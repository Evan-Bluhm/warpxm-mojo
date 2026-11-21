# 3D uniform block mesh.
#
# Tensor-product extension of `BlockMesh2D`. Hex elements numbered
#   e = k·nx·ny + j·nx + i
# for (i, j, k) ∈ [0, nx) × [0, ny) × [0, nz).
#
# Nodes per element: (P+1)³. Stored row-major in (ξ, η, ζ):
#   local n = kn·(P+1)² + jn·(P+1) + in_   for (in, jn, kn) ∈ [0, P+1)³
#
# Faces: three axis families. For each axis, periodic adds nx·ny·nz faces;
# non-periodic adds one extra (boundary) face per row.
#
# Per-element face table `elem_face_dir[e * 6 + dir]`:
#   dir 0 = LEFT   (axis 0, negative side)
#   dir 1 = RIGHT  (axis 0, positive side)
#   dir 2 = BOTTOM (axis 1, negative side)
#   dir 3 = TOP    (axis 1, positive side)
#   dir 4 = DOWN   (axis 2, negative side)
#   dir 5 = UP     (axis 2, positive side)
#
# Boundary faces have `face_neg_elem` = `GHOST_NEG` (for outward-facing
# negative side) or `face_pos_elem` = `GHOST_POS` (positive side), and a
# non-INTERIOR `face_bc_buf` entry. Spatial kernels detect these sentinels
# and synthesize ghost states via the BC dispatch.

from max.gpu.host import DeviceBuffer, DeviceContext
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT
from warpxm.core.types import Real, INDEX_DTYPE

# Negative sentinels — shared with mesh.block_2d for consistency.
comptime GHOST_NEG_3D: Int32 = -1
comptime GHOST_POS_3D: Int32 = -2

# Local face direction indices.
comptime FACE_DIR_LEFT_3D: Int = 0
comptime FACE_DIR_RIGHT_3D: Int = 1
comptime FACE_DIR_BOTTOM_3D: Int = 2
comptime FACE_DIR_TOP_3D: Int = 3
comptime FACE_DIR_DOWN_3D: Int = 4
comptime FACE_DIR_UP_3D: Int = 5


struct BlockMesh3D[P: Int]:
    """Uniform 3D DG hex mesh with (P+1)³ GLL nodes per element."""

    comptime num_nodes_per_elem: Int = (Self.P + 1) ** 3
    comptime num_faces_per_elem: Int = 6

    var nx: Int
    var ny: Int
    var nz: Int

    var num_elements: Int
    """nx · ny · nz."""

    var num_faces_axis0: Int
    var num_faces_axis1: Int
    var num_faces_axis2: Int
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
    var jacobian_x: Real
    var jacobian_y: Real
    var jacobian_z: Real

    var node_xyz_buf: DeviceBuffer[Real.dtype]
    var face_neg_elem_buf: DeviceBuffer[INDEX_DTYPE]
    var face_pos_elem_buf: DeviceBuffer[INDEX_DTYPE]
    var elem_face_dir_buf: DeviceBuffer[INDEX_DTYPE]
    var face_bc_buf: DeviceBuffer[INDEX_DTYPE]

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real, x_max: Real,
        y_min: Real, y_max: Real,
        z_min: Real, z_max: Real,
        nx: Int, ny: Int, nz: Int,
        periodic_x: Bool = True,
        periodic_y: Bool = True,
        periodic_z: Bool = True,
        bc_left: Int = BC_COPYOUT, bc_right: Int = BC_COPYOUT,
        bc_bottom: Int = BC_COPYOUT, bc_top: Int = BC_COPYOUT,
        bc_down: Int = BC_COPYOUT, bc_up: Int = BC_COPYOUT,
    ) raises:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_elements = nx * ny * nz
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.periodic_z = periodic_z
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.dx = (x_max - x_min) / Real(nx)
        self.dy = (y_max - y_min) / Real(ny)
        self.dz = (z_max - z_min) / Real(nz)
        self.jacobian_x = self.dx * Real(0.5)
        self.jacobian_y = self.dy * Real(0.5)
        self.jacobian_z = self.dz * Real(0.5)

        var nx_faces = nx if periodic_x else nx + 1
        var ny_faces = ny if periodic_y else ny + 1
        var nz_faces = nz if periodic_z else nz + 1
        self.num_faces_axis0 = nx_faces * ny * nz
        self.num_faces_axis1 = nx * ny_faces * nz
        self.num_faces_axis2 = nx * ny * nz_faces
        self.num_faces = self.num_faces_axis0 + self.num_faces_axis1 + self.num_faces_axis2

        # --- node coordinates ---------------------------------------------------
        var P1 = Self.P + 1
        var ref_nodes = NodalBasis1D[Self.P].nodes()
        var total_nodes = self.num_elements * Self.num_nodes_per_elem
        var node_host = ctx.enqueue_create_host_buffer[Real.dtype](total_nodes * 3)
        for k in range(nz):
            var zl = z_min + Real(k) * self.dz
            for j in range(ny):
                var yl = y_min + Real(j) * self.dy
                for i in range(nx):
                    var xl = x_min + Real(i) * self.dx
                    var e = k * nx * ny + j * nx + i
                    for kn in range(P1):
                        var z = zl + (Real(1.0) + ref_nodes[kn]) * self.jacobian_z
                        for jn in range(P1):
                            var y = yl + (Real(1.0) + ref_nodes[jn]) * self.jacobian_y
                            for in_ in range(P1):
                                var x = xl + (Real(1.0) + ref_nodes[in_]) * self.jacobian_x
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var idx = (e * Self.num_nodes_per_elem + n) * 3
                                node_host[idx + 0] = x
                                node_host[idx + 1] = y
                                node_host[idx + 2] = z
        self.node_xyz_buf = ctx.enqueue_create_buffer[Real.dtype](total_nodes * 3)
        ctx.enqueue_copy(dst_buf=self.node_xyz_buf, src_buf=node_host)

        # --- face connectivity --------------------------------------------------
        var fn_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var fp_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var efd_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_elements * 6)
        var fbc_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        for f in range(self.num_faces):
            fbc_host[f] = Int32(BC_INTERIOR)

        # Axis 0 (ξ-normal): nx_faces · ny · nz faces. Face (i_face, j, k)
        # has neg side at (i_face - 1, j, k) and pos side at (i_face, j, k).
        for k in range(nz):
            for j in range(ny):
                for i_face in range(nx_faces):
                    var f_idx = k * (ny * nx_faces) + j * nx_faces + i_face
                    if periodic_x:
                        var neg_i = (i_face - 1 + nx) % nx
                        var pos_i = i_face
                        var neg_e = k * nx * ny + j * nx + neg_i
                        var pos_e = k * nx * ny + j * nx + pos_i
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[neg_e * 6 + FACE_DIR_RIGHT_3D] = Int32(f_idx)
                        efd_host[pos_e * 6 + FACE_DIR_LEFT_3D] = Int32(f_idx)
                    else:
                        if i_face == 0:
                            var pos_e = k * nx * ny + j * nx + 0
                            fn_host[f_idx] = GHOST_NEG_3D
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[pos_e * 6 + FACE_DIR_LEFT_3D] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_left)
                        elif i_face == nx:
                            var neg_e = k * nx * ny + j * nx + (nx - 1)
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = GHOST_POS_3D
                            efd_host[neg_e * 6 + FACE_DIR_RIGHT_3D] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_right)
                        else:
                            var neg_e = k * nx * ny + j * nx + (i_face - 1)
                            var pos_e = k * nx * ny + j * nx + i_face
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[neg_e * 6 + FACE_DIR_RIGHT_3D] = Int32(f_idx)
                            efd_host[pos_e * 6 + FACE_DIR_LEFT_3D] = Int32(f_idx)

        # Axis 1 (η-normal): nx · ny_faces · nz faces.
        var off_axis1 = self.num_faces_axis0
        for k in range(nz):
            for j_face in range(ny_faces):
                for i in range(nx):
                    var f_idx = off_axis1 + k * (nx * ny_faces) + j_face * nx + i
                    if periodic_y:
                        var neg_j = (j_face - 1 + ny) % ny
                        var pos_j = j_face
                        var neg_e = k * nx * ny + neg_j * nx + i
                        var pos_e = k * nx * ny + pos_j * nx + i
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[neg_e * 6 + FACE_DIR_TOP_3D] = Int32(f_idx)
                        efd_host[pos_e * 6 + FACE_DIR_BOTTOM_3D] = Int32(f_idx)
                    else:
                        if j_face == 0:
                            var pos_e = k * nx * ny + 0 * nx + i
                            fn_host[f_idx] = GHOST_NEG_3D
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[pos_e * 6 + FACE_DIR_BOTTOM_3D] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_bottom)
                        elif j_face == ny:
                            var neg_e = k * nx * ny + (ny - 1) * nx + i
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = GHOST_POS_3D
                            efd_host[neg_e * 6 + FACE_DIR_TOP_3D] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_top)
                        else:
                            var neg_e = k * nx * ny + (j_face - 1) * nx + i
                            var pos_e = k * nx * ny + j_face * nx + i
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[neg_e * 6 + FACE_DIR_TOP_3D] = Int32(f_idx)
                            efd_host[pos_e * 6 + FACE_DIR_BOTTOM_3D] = Int32(f_idx)

        # Axis 2 (ζ-normal): nx · ny · nz_faces faces.
        var off_axis2 = self.num_faces_axis0 + self.num_faces_axis1
        for k_face in range(nz_faces):
            for j in range(ny):
                for i in range(nx):
                    var f_idx = off_axis2 + k_face * (nx * ny) + j * nx + i
                    if periodic_z:
                        var neg_k = (k_face - 1 + nz) % nz
                        var pos_k = k_face
                        var neg_e = neg_k * nx * ny + j * nx + i
                        var pos_e = pos_k * nx * ny + j * nx + i
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[neg_e * 6 + FACE_DIR_UP_3D] = Int32(f_idx)
                        efd_host[pos_e * 6 + FACE_DIR_DOWN_3D] = Int32(f_idx)
                    else:
                        if k_face == 0:
                            var pos_e = 0 * nx * ny + j * nx + i
                            fn_host[f_idx] = GHOST_NEG_3D
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[pos_e * 6 + FACE_DIR_DOWN_3D] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_down)
                        elif k_face == nz:
                            var neg_e = (nz - 1) * nx * ny + j * nx + i
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = GHOST_POS_3D
                            efd_host[neg_e * 6 + FACE_DIR_UP_3D] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_up)
                        else:
                            var neg_e = (k_face - 1) * nx * ny + j * nx + i
                            var pos_e = k_face * nx * ny + j * nx + i
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[neg_e * 6 + FACE_DIR_UP_3D] = Int32(f_idx)
                            efd_host[pos_e * 6 + FACE_DIR_DOWN_3D] = Int32(f_idx)

        self.face_neg_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.face_pos_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.elem_face_dir_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_elements * 6)
        self.face_bc_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        ctx.enqueue_copy(dst_buf=self.face_neg_elem_buf, src_buf=fn_host)
        ctx.enqueue_copy(dst_buf=self.face_pos_elem_buf, src_buf=fp_host)
        ctx.enqueue_copy(dst_buf=self.elem_face_dir_buf, src_buf=efd_host)
        ctx.enqueue_copy(dst_buf=self.face_bc_buf, src_buf=fbc_host)
