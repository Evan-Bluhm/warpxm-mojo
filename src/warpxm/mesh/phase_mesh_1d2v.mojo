# Uniform 1D2V phase-space block mesh on the GPU.
#
# Geometry + face connectivity for the 1D2V phase-space DG kernel. Mirrors
# `PhaseMesh1D1V[P_x, P_v]` (`mesh/phase_mesh_1d1v.mojo`) extended to three
# phase-space axes:
#
#   axis 0 = x  (physical, may be periodic)
#   axis 1 = v1 (= v_x, always truncated outflow at ±v1_max)
#   axis 2 = v2 (= v_y, always truncated outflow at ±v2_max)
#
# Phase elements are indexed in the same convention as
# `PhaseState1D2V` — v2 innermost:
#     phase_e = (ex · num_v1 + ev1) · num_v2 + ev2.
#
# Six local face directions per element, matching `BlockMesh[3, P]`:
#     elem_face_dir[phase_e * 6 + FACE_DIR_LEFT]   → x-negative
#     elem_face_dir[phase_e * 6 + FACE_DIR_RIGHT]  → x-positive
#     elem_face_dir[phase_e * 6 + FACE_DIR_BOTTOM] → v1-negative
#     elem_face_dir[phase_e * 6 + FACE_DIR_TOP]    → v1-positive
#     elem_face_dir[phase_e * 6 + FACE_DIR_DOWN]   → v2-negative
#     elem_face_dir[phase_e * 6 + FACE_DIR_UP]     → v2-positive
#
# Both velocity axes carry `BC_VLASOV_OUTFLOW` (zero-influx ghost state) at
# their truncation boundaries; the x-axis honours `periodic_x` with
# `BC_COPYOUT` (or another caller-supplied BC) on the boundary faces when
# non-periodic.

from max.gpu.host import DeviceBuffer, DeviceContext

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT, BC_VLASOV_OUTFLOW
from warpxm.core.types import Real, INDEX_DTYPE
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.mesh.block import (
    GHOST_NEG, GHOST_POS,
    FACE_DIR_LEFT, FACE_DIR_RIGHT,
    FACE_DIR_BOTTOM, FACE_DIR_TOP,
    FACE_DIR_DOWN, FACE_DIR_UP,
)


struct PhaseMesh1D2V[P_x: Int, P_v1: Int, P_v2: Int]:
    """Uniform 1D2V phase-space DG mesh with tensor-product GLL nodes."""

    comptime num_x_nodes_per_elem: Int = Self.P_x + 1
    comptime num_v1_nodes_per_elem: Int = Self.P_v1 + 1
    comptime num_v2_nodes_per_elem: Int = Self.P_v2 + 1
    comptime num_faces_per_elem: Int = 6

    var num_x: Int
    var num_v1: Int
    var num_v2: Int
    var num_phase_elements: Int
    """`num_x · num_v1 · num_v2`."""

    var periodic_x: Bool

    var x_min: Real
    var x_max: Real
    var v1_min: Real
    var v1_max: Real
    var v2_min: Real
    var v2_max: Real

    var dx: Real
    var dv1: Real
    var dv2: Real

    var jacobian_x: Real
    var jacobian_v1: Real
    var jacobian_v2: Real

    var num_x_faces: Int
    """x-axis face count: `num_x · num_v1 · num_v2` if `periodic_x`,
    `(num_x + 1) · num_v1 · num_v2` otherwise."""

    var num_v1_faces: Int
    """v1-axis face count, always `num_x · (num_v1 + 1) · num_v2`
    (v1 is non-periodic)."""

    var num_v2_faces: Int
    """v2-axis face count, always `num_x · num_v1 · (num_v2 + 1)`
    (v2 is non-periodic)."""

    var num_faces: Int
    """Total face count, `num_x_faces + num_v1_faces + num_v2_faces`."""

    var node_x_buf: DeviceBuffer[Real.dtype]
    """Physical x at every x-node, length `num_x · (P_x + 1)`."""

    var node_v1_buf: DeviceBuffer[Real.dtype]
    """Physical v1 at every v1-node, length `num_v1 · (P_v1 + 1)`."""

    var node_v2_buf: DeviceBuffer[Real.dtype]
    """Physical v2 at every v2-node, length `num_v2 · (P_v2 + 1)`."""

    var face_neg_elem_buf: DeviceBuffer[INDEX_DTYPE]
    """Phase-element index on the negative side of each face (or
    `GHOST_NEG` at a non-periodic boundary). Length `num_faces`."""

    var face_pos_elem_buf: DeviceBuffer[INDEX_DTYPE]
    """Phase-element index on the positive side of each face (or
    `GHOST_POS` at a non-periodic boundary). Length `num_faces`."""

    var elem_face_dir_buf: DeviceBuffer[INDEX_DTYPE]
    """Per-element face indices: `elem_face_dir[phase_e * 6 + dir]`."""

    var face_bc_buf: DeviceBuffer[INDEX_DTYPE]
    """BC kind per face (see `core/bc.mojo`). `BC_INTERIOR` on interior
    faces; the supplied `bc_left`/`bc_right` on x-boundary faces when
    `periodic_x` is False; `BC_VLASOV_OUTFLOW` on every v1- and v2-axis
    boundary face."""

    def __init__(
        out self,
        var ctx: DeviceContext,
        x_min: Real, x_max: Real,
        v1_min: Real, v1_max: Real,
        v2_min: Real, v2_max: Real,
        num_x: Int, num_v1: Int, num_v2: Int,
        periodic_x: Bool = True,
        bc_left: Int = BC_COPYOUT,
        bc_right: Int = BC_COPYOUT,
    ) raises:
        self.num_x = num_x
        self.num_v1 = num_v1
        self.num_v2 = num_v2
        self.num_phase_elements = num_x * num_v1 * num_v2
        self.periodic_x = periodic_x
        self.x_min = x_min
        self.x_max = x_max
        self.v1_min = v1_min
        self.v1_max = v1_max
        self.v2_min = v2_min
        self.v2_max = v2_max
        self.dx = (x_max - x_min) / Real(num_x)
        self.dv1 = (v1_max - v1_min) / Real(num_v1)
        self.dv2 = (v2_max - v2_min) / Real(num_v2)
        self.jacobian_x = self.dx * Real(0.5)
        self.jacobian_v1 = self.dv1 * Real(0.5)
        self.jacobian_v2 = self.dv2 * Real(0.5)

        var nx_faces_per_row = num_x if periodic_x else num_x + 1
        var nv1_faces_per_col = num_v1 + 1
        var nv2_faces_per_col = num_v2 + 1
        self.num_x_faces = nx_faces_per_row * num_v1 * num_v2
        self.num_v1_faces = num_x * nv1_faces_per_col * num_v2
        self.num_v2_faces = num_x * num_v1 * nv2_faces_per_col
        self.num_faces = self.num_x_faces + self.num_v1_faces + self.num_v2_faces

        # --- x-axis node coordinates ------------------------------------------
        var nx_per_elem = Self.num_x_nodes_per_elem
        var ref_x = NodalBasis1D[Self.P_x].nodes()
        var total_x = num_x * nx_per_elem
        var node_x_host = ctx.enqueue_create_host_buffer[Real.dtype](total_x)
        for e in range(num_x):
            var xl = x_min + Real(e) * self.dx
            for i in range(nx_per_elem):
                node_x_host[e * nx_per_elem + i] = (
                    xl + (Real(1.0) + ref_x[i]) * self.jacobian_x
                )
        self.node_x_buf = ctx.enqueue_create_buffer[Real.dtype](total_x)
        ctx.enqueue_copy(dst_buf=self.node_x_buf, src_buf=node_x_host)

        # --- v1-axis node coordinates -----------------------------------------
        var nv1_per_elem = Self.num_v1_nodes_per_elem
        var ref_v1 = NodalBasis1D[Self.P_v1].nodes()
        var total_v1 = num_v1 * nv1_per_elem
        var node_v1_host = ctx.enqueue_create_host_buffer[Real.dtype](total_v1)
        for e in range(num_v1):
            var vl = v1_min + Real(e) * self.dv1
            for i in range(nv1_per_elem):
                node_v1_host[e * nv1_per_elem + i] = (
                    vl + (Real(1.0) + ref_v1[i]) * self.jacobian_v1
                )
        self.node_v1_buf = ctx.enqueue_create_buffer[Real.dtype](total_v1)
        ctx.enqueue_copy(dst_buf=self.node_v1_buf, src_buf=node_v1_host)

        # --- v2-axis node coordinates -----------------------------------------
        var nv2_per_elem = Self.num_v2_nodes_per_elem
        var ref_v2 = NodalBasis1D[Self.P_v2].nodes()
        var total_v2 = num_v2 * nv2_per_elem
        var node_v2_host = ctx.enqueue_create_host_buffer[Real.dtype](total_v2)
        for e in range(num_v2):
            var vl = v2_min + Real(e) * self.dv2
            for i in range(nv2_per_elem):
                node_v2_host[e * nv2_per_elem + i] = (
                    vl + (Real(1.0) + ref_v2[i]) * self.jacobian_v2
                )
        self.node_v2_buf = ctx.enqueue_create_buffer[Real.dtype](total_v2)
        ctx.enqueue_copy(dst_buf=self.node_v2_buf, src_buf=node_v2_host)

        # --- face connectivity ------------------------------------------------
        var fn_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var fp_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        var efd_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](
            self.num_phase_elements * 6
        )
        var fbc_host = ctx.enqueue_create_host_buffer[INDEX_DTYPE](self.num_faces)
        for k in range(self.num_phase_elements * 6):
            efd_host[k] = GHOST_NEG
        for f in range(self.num_faces):
            fbc_host[f] = Int32(BC_INTERIOR)

        # x-axis faces: indexed by (i_face, ev1, ev2) with i_face outer to
        # match the phase_e convention (ex outer). f_idx layout mirrors
        # `phase_e = ex * num_v1 * num_v2 + ev1 * num_v2 + ev2`.
        for i_face in range(nx_faces_per_row):
            for ev1 in range(num_v1):
                for ev2 in range(num_v2):
                    var f_idx = (
                        i_face * (num_v1 * num_v2) + ev1 * num_v2 + ev2
                    )
                    if periodic_x:
                        var neg_ex = (i_face - 1 + num_x) % num_x
                        var pos_ex = i_face
                        var neg_e = (neg_ex * num_v1 + ev1) * num_v2 + ev2
                        var pos_e = (pos_ex * num_v1 + ev1) * num_v2 + ev2
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[neg_e * 6 + FACE_DIR_RIGHT] = Int32(f_idx)
                        efd_host[pos_e * 6 + FACE_DIR_LEFT] = Int32(f_idx)
                    else:
                        if i_face == 0:
                            var pos_e = (0 * num_v1 + ev1) * num_v2 + ev2
                            fn_host[f_idx] = GHOST_NEG
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[pos_e * 6 + FACE_DIR_LEFT] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_left)
                        elif i_face == num_x:
                            var neg_e = (
                                ((num_x - 1) * num_v1 + ev1) * num_v2 + ev2
                            )
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = GHOST_POS
                            efd_host[neg_e * 6 + FACE_DIR_RIGHT] = Int32(f_idx)
                            fbc_host[f_idx] = Int32(bc_right)
                        else:
                            var neg_ex = i_face - 1
                            var pos_ex = i_face
                            var neg_e = (neg_ex * num_v1 + ev1) * num_v2 + ev2
                            var pos_e = (pos_ex * num_v1 + ev1) * num_v2 + ev2
                            fn_host[f_idx] = Int32(neg_e)
                            fp_host[f_idx] = Int32(pos_e)
                            efd_host[neg_e * 6 + FACE_DIR_RIGHT] = Int32(f_idx)
                            efd_host[pos_e * 6 + FACE_DIR_LEFT] = Int32(f_idx)

        # v1-axis faces (non-periodic). Indexed by (ex, j_face, ev2) where
        # j_face ∈ [0, num_v1]. Offset starts at `num_x_faces`.
        var off_v1 = self.num_x_faces
        for ex in range(num_x):
            for j_face in range(nv1_faces_per_col):
                for ev2 in range(num_v2):
                    var f_idx = (
                        off_v1
                        + ex * (nv1_faces_per_col * num_v2)
                        + j_face * num_v2
                        + ev2
                    )
                    if j_face == 0:
                        var pos_e = (ex * num_v1 + 0) * num_v2 + ev2
                        fn_host[f_idx] = GHOST_NEG
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[pos_e * 6 + FACE_DIR_BOTTOM] = Int32(f_idx)
                        fbc_host[f_idx] = Int32(BC_VLASOV_OUTFLOW)
                    elif j_face == num_v1:
                        var neg_e = (
                            (ex * num_v1 + (num_v1 - 1)) * num_v2 + ev2
                        )
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = GHOST_POS
                        efd_host[neg_e * 6 + FACE_DIR_TOP] = Int32(f_idx)
                        fbc_host[f_idx] = Int32(BC_VLASOV_OUTFLOW)
                    else:
                        var neg_ev1 = j_face - 1
                        var pos_ev1 = j_face
                        var neg_e = (ex * num_v1 + neg_ev1) * num_v2 + ev2
                        var pos_e = (ex * num_v1 + pos_ev1) * num_v2 + ev2
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[neg_e * 6 + FACE_DIR_TOP] = Int32(f_idx)
                        efd_host[pos_e * 6 + FACE_DIR_BOTTOM] = Int32(f_idx)

        # v2-axis faces (non-periodic). Indexed by (ex, ev1, k_face) where
        # k_face ∈ [0, num_v2]. Offset starts at `num_x_faces + num_v1_faces`.
        var off_v2 = self.num_x_faces + self.num_v1_faces
        for ex in range(num_x):
            for ev1 in range(num_v1):
                for k_face in range(nv2_faces_per_col):
                    var f_idx = (
                        off_v2
                        + ex * (num_v1 * nv2_faces_per_col)
                        + ev1 * nv2_faces_per_col
                        + k_face
                    )
                    if k_face == 0:
                        var pos_e = (ex * num_v1 + ev1) * num_v2 + 0
                        fn_host[f_idx] = GHOST_NEG
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[pos_e * 6 + FACE_DIR_DOWN] = Int32(f_idx)
                        fbc_host[f_idx] = Int32(BC_VLASOV_OUTFLOW)
                    elif k_face == num_v2:
                        var neg_e = (
                            (ex * num_v1 + ev1) * num_v2 + (num_v2 - 1)
                        )
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = GHOST_POS
                        efd_host[neg_e * 6 + FACE_DIR_UP] = Int32(f_idx)
                        fbc_host[f_idx] = Int32(BC_VLASOV_OUTFLOW)
                    else:
                        var neg_ev2 = k_face - 1
                        var pos_ev2 = k_face
                        var neg_e = (ex * num_v1 + ev1) * num_v2 + neg_ev2
                        var pos_e = (ex * num_v1 + ev1) * num_v2 + pos_ev2
                        fn_host[f_idx] = Int32(neg_e)
                        fp_host[f_idx] = Int32(pos_e)
                        efd_host[neg_e * 6 + FACE_DIR_UP] = Int32(f_idx)
                        efd_host[pos_e * 6 + FACE_DIR_DOWN] = Int32(f_idx)

        self.face_neg_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](
            self.num_faces
        )
        self.face_pos_elem_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](
            self.num_faces
        )
        self.elem_face_dir_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](
            self.num_phase_elements * 6
        )
        self.face_bc_buf = ctx.enqueue_create_buffer[INDEX_DTYPE](self.num_faces)
        ctx.enqueue_copy(dst_buf=self.face_neg_elem_buf, src_buf=fn_host)
        ctx.enqueue_copy(dst_buf=self.face_pos_elem_buf, src_buf=fp_host)
        ctx.enqueue_copy(dst_buf=self.elem_face_dir_buf, src_buf=efd_host)
        ctx.enqueue_copy(dst_buf=self.face_bc_buf, src_buf=fbc_host)

    # -----------------------------------------------------------------
    # Host-side helpers: initial conditions and diagnostics. They own the
    # phase-node ordering and the buffer layout; component 0 (the
    # distribution function) is the one read and written.
    # -----------------------------------------------------------------

    def fill[NC: Int, IC: def(Real, Real, Real) -> Real](
        self,
        mut f: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, NC],
        ic: IC,
    ) raises:
        """Set the distribution function to `ic(x, v1, v2)` at every phase
        node."""
        var ref_x = NodalBasis1D[Self.P_x].nodes()
        var ref_v1 = NodalBasis1D[Self.P_v1].nodes()
        var ref_v2 = NodalBasis1D[Self.P_v2].nodes()
        comptime PX1 = Self.P_x + 1
        comptime PV11 = Self.P_v1 + 1
        comptime PV21 = Self.P_v2 + 1
        comptime NPE = PX1 * PV11 * PV21
        with f.buf.map_to_host() as h:
            for ex in range(self.num_x):
                var xl = self.x_min + Real(ex) * self.dx
                for ev1 in range(self.num_v1):
                    var v1l = self.v1_min + Real(ev1) * self.dv1
                    for ev2 in range(self.num_v2):
                        var v2l = self.v2_min + Real(ev2) * self.dv2
                        var phase_e = (ex * self.num_v1 + ev1) * self.num_v2 + ev2
                        for iv2 in range(PV21):
                            var v2 = v2l + (Real(1.0) + ref_v2[iv2]) * self.jacobian_v2
                            for iv1 in range(PV11):
                                var v1 = v1l + (Real(1.0) + ref_v1[iv1]) * self.jacobian_v1
                                for ix in range(PX1):
                                    var x = xl + (Real(1.0) + ref_x[ix]) * self.jacobian_x
                                    var n = (iv2 * PV11 + iv1) * PX1 + ix
                                    h[(phase_e * NPE + n) * NC] = ic(x, v1, v2)

    def visit_nodes[NC: Int, Visit: def(Real, Real, Real, Real, Real) -> None](
        self,
        mut f: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, NC],
        visit: Visit,
    ) raises:
        """Call `visit(x, v1, v2, w, f_node)` at every phase node. `w` is
        the phase-space quadrature weight."""
        var ref_x = NodalBasis1D[Self.P_x].nodes()
        var ref_v1 = NodalBasis1D[Self.P_v1].nodes()
        var ref_v2 = NodalBasis1D[Self.P_v2].nodes()
        var w_x = NodalBasis1D[Self.P_x].weights()
        var w_v1 = NodalBasis1D[Self.P_v1].weights()
        var w_v2 = NodalBasis1D[Self.P_v2].weights()
        comptime PX1 = Self.P_x + 1
        comptime PV11 = Self.P_v1 + 1
        comptime PV21 = Self.P_v2 + 1
        comptime NPE = PX1 * PV11 * PV21
        var w_jac = self.jacobian_x * self.jacobian_v1 * self.jacobian_v2
        with f.buf.map_to_host() as h:
            for ex in range(self.num_x):
                var xl = self.x_min + Real(ex) * self.dx
                for ev1 in range(self.num_v1):
                    var v1l = self.v1_min + Real(ev1) * self.dv1
                    for ev2 in range(self.num_v2):
                        var v2l = self.v2_min + Real(ev2) * self.dv2
                        var phase_e = (ex * self.num_v1 + ev1) * self.num_v2 + ev2
                        for iv2 in range(PV21):
                            var v2 = v2l + (Real(1.0) + ref_v2[iv2]) * self.jacobian_v2
                            for iv1 in range(PV11):
                                var v1 = v1l + (Real(1.0) + ref_v1[iv1]) * self.jacobian_v1
                                for ix in range(PX1):
                                    var x = xl + (Real(1.0) + ref_x[ix]) * self.jacobian_x
                                    var n = (iv2 * PV11 + iv1) * PX1 + ix
                                    visit(
                                        x, v1, v2,
                                        w_x[ix] * w_v1[iv1] * w_v2[iv2] * w_jac,
                                        h[(phase_e * NPE + n) * NC],
                                    )
