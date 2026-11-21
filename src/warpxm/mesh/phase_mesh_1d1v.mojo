# Uniform 1D1V phase-space block mesh on the GPU.
#
# Mirrors the configuration-space `BlockMesh1D[P]` but carries both a
# physical-space axis `x ∈ [x_min, x_max]` and a velocity-space axis
# `v ∈ [v_min, v_max]`. Each phase-space cell is a tensor product of one
# physical-space cell and one velocity-space cell.
#
# Phase elements are indexed row-major with velocity inner:
#     phase_e = ex * num_v + ev    for (ex, ev) ∈ [0, num_x) × [0, num_v).
#
# Per-element local face directions follow the upstream / `BlockMesh2D`
# layout — left/right along x, bottom/top along v:
#     elem_face_dir[phase_e * 4 + FACE_DIR_LEFT]   → x-negative face
#     elem_face_dir[phase_e * 4 + FACE_DIR_RIGHT]  → x-positive face
#     elem_face_dir[phase_e * 4 + FACE_DIR_BOTTOM] → v-negative face
#     elem_face_dir[phase_e * 4 + FACE_DIR_TOP]    → v-positive face
#
# The v-axis is always treated as non-periodic — `f` is truncated to zero
# outside `[v_min, v_max]`. The x-axis can be periodic or non-periodic.

from max.gpu.host import DeviceBuffer, DeviceContext

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT, BC_VLASOV_OUTFLOW
from warpxm.core.types import Real, INDEX_DTYPE
from warpxm.mesh.block_2d import (
    GHOST_NEG, GHOST_POS,
    FACE_DIR_LEFT, FACE_DIR_RIGHT, FACE_DIR_BOTTOM, FACE_DIR_TOP,
)


struct PhaseMesh1D1V[P_x: Int, P_v: Int]:
    """Uniform 1D1V phase-space DG mesh with tensor-product GLL nodes."""

    comptime num_x_nodes_per_elem: Int = Self.P_x + 1
    comptime num_v_nodes_per_elem: Int = Self.P_v + 1

    var num_x: Int
    """Number of physical-space elements."""

    var num_v: Int
    """Number of velocity-space elements."""

    var num_phase_elements: Int
    """`num_x * num_v`."""

    var periodic_x: Bool
    """Whether the physical-space axis wraps around. Velocity-space is
    always treated as non-periodic — `f` is truncated to zero outside
    `[v_min, v_max]`."""

    var x_min: Real
    var x_max: Real
    var v_min: Real
    var v_max: Real

    var dx: Real
    """Per-element width along x, equal to `(x_max - x_min) / num_x`."""

    var dv: Real
    """Per-element width along v, equal to `(v_max - v_min) / num_v`."""

    var jacobian_x: Real
    """Reference-to-physical Jacobian along x, equal to `dx / 2`."""

    var jacobian_v: Real
    """Reference-to-physical Jacobian along v, equal to `dv / 2`."""

    var num_x_faces: Int
    """x-axis face count. `num_x * num_v` if periodic, `(num_x + 1) * num_v`
    otherwise."""

    var num_v_faces: Int
    """v-axis face count, always `num_x * (num_v + 1)` (v is non-periodic)."""

    var num_faces: Int
    """Total face count, `num_x_faces + num_v_faces`."""

    var node_x_buf: DeviceBuffer[Real.dtype]
    """Physical x at every x-node, length `num_x * (P_x + 1)`. The
    coordinate is independent of velocity element, so we store it once
    per x-axis node rather than once per phase node."""

    var node_v_buf: DeviceBuffer[Real.dtype]
    """Physical v at every v-node, length `num_v * (P_v + 1)`. Same
    storage choice: v is independent of physical element."""

    var face_neg_elem_buf: DeviceBuffer[INDEX_DTYPE]
    """Phase-element index on the negative side of each face (or
    `GHOST_NEG` at a non-periodic boundary). Length `num_faces`."""

    var face_pos_elem_buf: DeviceBuffer[INDEX_DTYPE]
    """Phase-element index on the positive side of each face (or
    `GHOST_POS` at a non-periodic boundary). Length `num_faces`."""

    var elem_face_dir_buf: DeviceBuffer[INDEX_DTYPE]
    """Per-element face indices: `elem_face_dir[phase_e * 4 + dir]`."""

    var face_bc_buf: DeviceBuffer[INDEX_DTYPE]
    """BC kind per face (see `core/bc.mojo`). `BC_INTERIOR` on interior
    faces; `bc_left`/`bc_right` on the x-boundary faces when `periodic_x`
    is False; `BC_COPYOUT` on the v-boundary faces."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real, x_max: Real, v_min: Real, v_max: Real,
        num_x: Int, num_v: Int,
        periodic_x: Bool = True,
        bc_left: Int = BC_COPYOUT,
        bc_right: Int = BC_COPYOUT,
    ) raises:
        """Allocate phase mesh geometry + face connectivity. Velocity
        axis is always non-periodic with `BC_COPYOUT` (zero-influx) at
        `v = v_min` and `v = v_max`; the x-axis honours `periodic_x`."""
        self.num_x = num_x
        self.num_v = num_v
        self.num_phase_elements = num_x * num_v
        self.periodic_x = periodic_x
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max
        self.dx = (x_max - x_min) / Real(num_x)
        self.dv = (v_max - v_min) / Real(num_v)
        self.jacobian_x = self.dx * Real(0.5)
        self.jacobian_v = self.dv * Real(0.5)

        self.num_x_faces = num_x * num_v if periodic_x else (num_x + 1) * num_v
        self.num_v_faces = num_x * (num_v + 1)
        self.num_faces = self.num_x_faces + self.num_v_faces

        # --- x-axis node coordinates ------------------------------------------
        var x_nodes_per_elem = Self.num_x_nodes_per_elem
        var ref_x_nodes = NodalBasis1D[Self.P_x].nodes()
        var total_x_nodes = num_x * x_nodes_per_elem

        var node_x_host = ctx.enqueue_create_host_buffer[Real.dtype](total_x_nodes)
        for e in range(num_x):
            var xl = x_min + Real(e) * self.dx
            for i in range(x_nodes_per_elem):
                var xi = ref_x_nodes[i]
                node_x_host[e * x_nodes_per_elem + i] = (
                    xl + (Real(1.0) + xi) * self.jacobian_x
                )

        self.node_x_buf = ctx.enqueue_create_buffer[Real.dtype](total_x_nodes)
        ctx.enqueue_copy(dst_buf=self.node_x_buf, src_buf=node_x_host)

        # --- v-axis node coordinates ------------------------------------------
        var v_nodes_per_elem = Self.num_v_nodes_per_elem
        var ref_v_nodes = NodalBasis1D[Self.P_v].nodes()
        var total_v_nodes = num_v * v_nodes_per_elem

        var node_v_host = ctx.enqueue_create_host_buffer[Real.dtype](total_v_nodes)
        for e in range(num_v):
            var vl = v_min + Real(e) * self.dv
            for i in range(v_nodes_per_elem):
                var xi = ref_v_nodes[i]
                node_v_host[e * v_nodes_per_elem + i] = (
                    vl + (Real(1.0) + xi) * self.jacobian_v
                )

        self.node_v_buf = ctx.enqueue_create_buffer[Real.dtype](total_v_nodes)
        ctx.enqueue_copy(dst_buf=self.node_v_buf, src_buf=node_v_host)

        # --- face connectivity ------------------------------------------------
        var fn_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var fp_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var efd_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](
            self.num_phase_elements * 4
        )
        for k in range(self.num_phase_elements * 4):
            efd_host[k] = GHOST_NEG

        # x-axis faces: indexed by (i_face, ev) where i_face ∈ [0, num_x)
        # for periodic, [0, num_x] for non-periodic.
        var nx_faces_per_row = num_x if periodic_x else num_x + 1
        for ev in range(num_v):
            for i_face in range(nx_faces_per_row):
                var f_idx = ev * nx_faces_per_row + i_face
                if periodic_x:
                    var neg_ex = (i_face - 1 + num_x) % num_x
                    var pos_ex = i_face
                    fn_host[f_idx] = Int32(neg_ex * num_v + ev)
                    fp_host[f_idx] = Int32(pos_ex * num_v + ev)
                    efd_host[(neg_ex * num_v + ev) * 4 + FACE_DIR_RIGHT] = Int32(f_idx)
                    efd_host[(pos_ex * num_v + ev) * 4 + FACE_DIR_LEFT] = Int32(f_idx)
                else:
                    if i_face == 0:
                        fn_host[f_idx] = GHOST_NEG
                        fp_host[f_idx] = Int32(0 * num_v + ev)
                        efd_host[(0 * num_v + ev) * 4 + FACE_DIR_LEFT] = Int32(f_idx)
                    elif i_face == num_x:
                        fn_host[f_idx] = Int32((num_x - 1) * num_v + ev)
                        fp_host[f_idx] = GHOST_POS
                        efd_host[((num_x - 1) * num_v + ev) * 4 + FACE_DIR_RIGHT] = Int32(f_idx)
                    else:
                        var neg_ex = i_face - 1
                        var pos_ex = i_face
                        fn_host[f_idx] = Int32(neg_ex * num_v + ev)
                        fp_host[f_idx] = Int32(pos_ex * num_v + ev)
                        efd_host[(neg_ex * num_v + ev) * 4 + FACE_DIR_RIGHT] = Int32(f_idx)
                        efd_host[(pos_ex * num_v + ev) * 4 + FACE_DIR_LEFT] = Int32(f_idx)

        # v-axis faces: always non-periodic. Indexed by (ex, j_face) where
        # j_face ∈ [0, num_v]. Offset starts at `self.num_x_faces`.
        for j_face in range(num_v + 1):
            for ex in range(num_x):
                var f_idx = self.num_x_faces + j_face * num_x + ex
                if j_face == 0:
                    fn_host[f_idx] = GHOST_NEG
                    fp_host[f_idx] = Int32(ex * num_v + 0)
                    efd_host[(ex * num_v + 0) * 4 + FACE_DIR_BOTTOM] = Int32(f_idx)
                elif j_face == num_v:
                    fn_host[f_idx] = Int32(ex * num_v + (num_v - 1))
                    fp_host[f_idx] = GHOST_POS
                    efd_host[(ex * num_v + (num_v - 1)) * 4 + FACE_DIR_TOP] = Int32(f_idx)
                else:
                    var neg_ev = j_face - 1
                    var pos_ev = j_face
                    fn_host[f_idx] = Int32(ex * num_v + neg_ev)
                    fp_host[f_idx] = Int32(ex * num_v + pos_ev)
                    efd_host[(ex * num_v + neg_ev) * 4 + FACE_DIR_TOP] = Int32(f_idx)
                    efd_host[(ex * num_v + pos_ev) * 4 + FACE_DIR_BOTTOM] = Int32(f_idx)

        self.face_neg_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.face_pos_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        self.elem_face_dir_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](
            self.num_phase_elements * 4
        )
        ctx.enqueue_copy(dst_buf=self.face_neg_elem_buf, src_buf=fn_host)
        ctx.enqueue_copy(dst_buf=self.face_pos_elem_buf, src_buf=fp_host)
        ctx.enqueue_copy(dst_buf=self.elem_face_dir_buf, src_buf=efd_host)

        # --- BC kind per face -------------------------------------------------
        var face_bc_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        for f in range(self.num_faces):
            face_bc_host[f] = Int32(BC_INTERIOR)
        if not periodic_x:
            # Leftmost / rightmost x-faces in each velocity row.
            for ev in range(num_v):
                face_bc_host[ev * nx_faces_per_row + 0] = Int32(bc_left)
                face_bc_host[ev * nx_faces_per_row + num_x] = Int32(bc_right)
        # v-axis boundary faces: zero-influx truncation. The physical
        # distribution function vanishes outside [v_min, v_max]; the
        # kernel's BC dispatch will read this and synthesise a zero
        # ghost state, so the upwind flux cleanly enforces no inflow.
        for ex in range(num_x):
            face_bc_host[self.num_x_faces + 0 * num_x + ex] = Int32(BC_VLASOV_OUTFLOW)
            face_bc_host[self.num_x_faces + num_v * num_x + ex] = Int32(BC_VLASOV_OUTFLOW)
        self.face_bc_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        ctx.enqueue_copy(dst_buf=self.face_bc_buf, src_buf=face_bc_host)
