# 1D1V nodal DG spatial solver for the kinetic Vlasov-Maxwell equation.
#
# Mirrors upstream `src/apps/vlasovmaxwell/vlasovmaxwell_kokkos.cc`
# specialised to one physical-space and one velocity-space dimension
# (1D1V). The unknown is `f(x, v, t)`; the kernel evaluates the DG
# residual
#
#     ∂f/∂t = -∂(v f)/∂x - ∂(a_v f)/∂v + surface contributions
#
# at every tensor-product GLL phase node, with the wave speed along the
# x-axis taken from each phase node's local `v`-coordinate. The
# velocity-direction Lorentz acceleration `a_v` couples to Maxwell aux
# state; for this slice it is hard-coded to zero (free streaming). The
# next slice will read `a_v` from an aux buffer.
#
# Structurally the phase mesh is a tensor-product 2D mesh with axis-0 =
# physical x and axis-1 = velocity v; the kernel layout therefore
# closely follows `spatial/dg_core_2d.mojo`. The two differences are
#   * the wave speed for the x-axis surface flux is the local
#     v-coordinate (position-dependent) rather than a state-derived
#     quantity, and
#   * the v-axis surface flux uses `a_v`, which is `0` for this slice.

from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

from warpxm.apps.vlasovmaxwell import (
    VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC, AUX_EX,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_VLASOV_OUTFLOW
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import (
    GHOST_NEG, GHOST_POS,
    FACE_DIR_LEFT, FACE_DIR_RIGHT, FACE_DIR_BOTTOM, FACE_DIR_TOP,
)
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


@always_inline
def _load_phase_node[
    P_x: Int, P_v: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    phase_e: Int,
    local_node: Int,
) -> Array[Real, NC]:
    comptime assert q.flat_rank == 1
    comptime nodes_per_phase = (P_x + 1) * (P_v + 1)
    var base = (phase_e * nodes_per_phase + local_node) * NC
    var out: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out^


def _vlasov_rhs_kernel_1d1v[
    P_x: Int, P_v: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
    NodeVLayout: TensorLayout, AuxLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    node_v: TileTensor[WXM_DTYPE, NodeVLayout, MutAnyOrigin],
    aux: TileTensor[WXM_DTYPE, AuxLayout, MutAnyOrigin],
    num_v_i64: Int64,
    accel_factor: Real,
    inv_jacobian_x: Real, inv_jacobian_v: Real,
    num_phase_elements_i64: Int64,
):
    """Per-phase-node DG residual for 1D1V Vlasov.

    The Lorentz acceleration `a_v = (q/m)/λ · E_x` is read from the
    Maxwell aux buffer (flat per-x-node storage of
    `(E_x, E_y, E_z, B_x, B_y, B_z)`); for 1D1V only `E_x` enters
    because `v_y = v_z = 0` collapses `v × B` to zero.

    Thread layout: `(block_idx.x, block_idx.y) = (ev, ex)`,
    `(thread_idx.x, thread_idx.y) = (ix, iv)` indexing the local node
    `(ix, iv) ∈ [0, P_x+1) × [0, P_v+1)`.
    """
    var num_v = Int(num_v_i64)
    var num_phase_elements = Int(num_phase_elements_i64)
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime nodes_per_phase = P1x * P1v
    comptime NC = VLASOV_NC
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1
    comptime assert face_bc.flat_rank == 1
    comptime assert node_v.flat_rank == 1
    comptime assert aux.flat_rank == 1

    # The launch grid is (num_v, num_x), so the element indices come
    # straight from the block index (no 64-bit divide/modulo per thread).
    var ev = Int(block_idx.x)
    var ex = Int(block_idx.y)
    var phase_e = ex * num_v + ev
    var ix = Int(thread_idx.x)
    var iv = Int(thread_idx.y)
    if phase_e >= num_phase_elements or ix >= P1x or iv >= P1v:
        return

    var local_n = iv * P1x + ix

    # v-coordinate for the current phase node — same on both sides of an
    # x-axis face (the face shares ev between negative and positive cells).
    var vx_n: Real = rebind[Scalar[WXM_DTYPE]](node_v[ev * P1v + iv])

    # Lorentz acceleration along v from the Maxwell aux state. In 1D1V
    # the aux is indexed by physical-space x-node only; both v-faces of
    # a phase cell see the same `a_v` (since aux doesn't depend on `v`)
    # so we can use `av_n` as the wave speed on every v-direction face.
    var node_x = ex * P1x + ix
    var ex_field: Real = rebind[Scalar[WXM_DTYPE]](
        aux[node_x * VLASOV_AUX_NC + AUX_EX]
    )
    var av_n: Real = VlasovMaxwell.acceleration_v_1d1v_static(
        ex_field, accel_factor, Real(1.0),
    )

    # Shared-memory cache of the per-node fluxes for this phase cell, so
    # the volume sweep can re-read neighbour-node fluxes without going
    # back to global memory.
    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[P1x * P1v, NC, 2]())

    var f_n = _load_phase_node[P_x, P_v, NC](q, phase_e, local_n)
    var fx_n = VlasovMaxwell.internal_flux_x_static[NC](f_n, vx_n)
    var fv_n = VlasovMaxwell.internal_flux_v_static[NC](f_n, av_n)
    comptime for c in range(NC):
        f_shared[local_n, c, 0] = rebind[f_shared.ElementType](fx_n[c])
        f_shared[local_n, c, 1] = rebind[f_shared.ElementType](fv_n[c])

    barrier()

    # FP64 and local-memory hygiene (see CLAUDE.md, "Kernel performance
    # rules"): the launcher passes 1 / jacobian and (q/m) / λ_skin, the
    # weights enter as comptime reciprocals, and the differentiation
    # matrices are read through `diff_entry` — so the kernel issues no
    # double divisions and keeps nothing in per-thread local memory.
    # --- Volume term ------------------------------------------------------
    # ∂F_x/∂x along the ix axis (uses the P_x diff matrix);
    # ∂F_v/∂v along the iv axis (uses the P_v diff matrix).
    var dfdx: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))
    var dgdv: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))

    comptime for kn in range(P1x):
        var dx_ix_kn: Scalar[WXM_DTYPE] = NodalBasis1D[P_x].diff_entry[kn](ix)
        var k_node = iv * P1x + kn
        comptime for c in range(NC):
            var fx_kn = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 0])
            dfdx[c] += dx_ix_kn * fx_kn

    comptime for ln in range(P1v):
        var dv_iv_ln: Scalar[WXM_DTYPE] = NodalBasis1D[P_v].diff_entry[ln](iv)
        var l_node = ln * P1x + ix
        comptime for c in range(NC):
            var fv_ln = rebind[Scalar[WXM_DTYPE]](f_shared[l_node, c, 1])
            dgdv[c] += dv_iv_ln * fv_ln

    # --- Surface term -----------------------------------------------------
    # Only the four boundary nodes of the phase cell contribute. We do
    # not apply BC closures yet — the v-axis copyout fallthrough already
    # gives a zero ghost state for slice 3 (interior `f` carried through
    # an inert numerical flux because `a_v = 0`), and the x-axis is
    # periodic in the smoke test that uses this kernel.
    comptime inv_wx_0: Real = Real(1.0) / NodalBasis1D[P_x].weights()[0]
    var sx_0: Scalar[WXM_DTYPE] = inv_jacobian_x * inv_wx_0
    comptime inv_wx_P: Real = Real(1.0) / NodalBasis1D[P_x].weights()[P_x]
    var sx_P: Scalar[WXM_DTYPE] = inv_jacobian_x * inv_wx_P
    comptime inv_wv_0: Real = Real(1.0) / NodalBasis1D[P_v].weights()[0]
    var sv_0: Scalar[WXM_DTYPE] = inv_jacobian_v * inv_wv_0
    comptime inv_wv_P: Real = Real(1.0) / NodalBasis1D[P_v].weights()[P_v]
    var sv_P: Scalar[WXM_DTYPE] = inv_jacobian_v * inv_wv_P

    var surface: Array[Real, NC] = Array[Real, NC](fill=Real(0.0))

    if ix == 0:
        var fL = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[phase_e * 4 + FACE_DIR_LEFT]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fL]))
        var q_neg: Array[Real, NC]
        if neg_e == Int(GHOST_NEG):
            q_neg = f_n.copy()  # copyout fallback
        else:
            q_neg = _load_phase_node[P_x, P_v, NC](q, neg_e, iv * P1x + P_x)
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            q_neg, f_n, vx_n,
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fx_n[c]) * sx_0

    if ix == P_x:
        var fR = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[phase_e * 4 + FACE_DIR_RIGHT]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fR]))
        var q_pos: Array[Real, NC]
        if pos_e == Int(GHOST_POS):
            q_pos = f_n.copy()
        else:
            q_pos = _load_phase_node[P_x, P_v, NC](q, pos_e, iv * P1x + 0)
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            f_n, q_pos, vx_n,
        )
        comptime for c in range(NC):
            surface[c] += (fx_n[c] - f_star[c]) * sx_P

    if iv == 0:
        var fB = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[phase_e * 4 + FACE_DIR_BOTTOM]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fB]))
        var q_neg: Array[Real, NC]
        if neg_e == Int(GHOST_NEG):
            var bc = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fB]))
            if bc == BC_VLASOV_OUTFLOW:
                q_neg = Array[Real, NC](fill=Real(0.0))
            else:
                q_neg = f_n.copy()  # BC_COPYOUT fallback
        else:
            q_neg = _load_phase_node[P_x, P_v, NC](q, neg_e, P_v * P1x + ix)
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            q_neg, f_n, av_n,
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fv_n[c]) * sv_0

    if iv == P_v:
        var fT = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[phase_e * 4 + FACE_DIR_TOP]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fT]))
        var q_pos: Array[Real, NC]
        if pos_e == Int(GHOST_POS):
            var bc = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fT]))
            if bc == BC_VLASOV_OUTFLOW:
                q_pos = Array[Real, NC](fill=Real(0.0))
            else:
                q_pos = f_n.copy()
        else:
            q_pos = _load_phase_node[P_x, P_v, NC](q, pos_e, 0 * P1x + ix)
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            f_n, q_pos, av_n,
        )
        comptime for c in range(NC):
            surface[c] += (fv_n[c] - f_star[c]) * sv_P

    # --- Write back -------------------------------------------------------
    var base = (phase_e * nodes_per_phase + local_n) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = (
            -dfdx[c] * inv_jacobian_x - dgdv[c] * inv_jacobian_v + surface[c]
        )
        rhs[base + c] = rebind[rhs.ElementType](v)


struct DGVlasovSolver1D1V[P_x: Int, P_v: Int]:
    """Host-side launcher for the 1D1V Vlasov DG kernel.

    Reads `E_x` from a per-x-node Maxwell aux buffer to form the
    v-axis Lorentz acceleration `a_v = (q/m)/λ · E_x`. The aux buffer
    is a `DGState1D[P_x, VLASOV_AUX_NC]`; the driver populates it (in
    the future from an actual Maxwell solver, currently from a fixed
    user-supplied profile)."""

    var app: VlasovMaxwell

    def __init__(out self, app: VlasovMaxwell):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh1D1V[Self.P_x, Self.P_v],
        mut q: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        mut aux: DGState1D[Self.P_x, VLASOV_AUX_NC],
        mut rhs: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
    ) raises:
        comptime P1x = Self.P_x + 1
        comptime P1v = Self.P_v + 1

        var q_layout = row_major(Int(q.buf_len))
        var face_layout = row_major(Int(mesh.num_faces))
        var ef_layout = row_major(Int(mesh.num_phase_elements * 4))
        var node_v_layout = row_major(Int(mesh.num_v * P1v))
        var aux_layout = row_major(Int(aux.buf_len))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)
        var node_v_view = TileTensor(mesh.node_v_buf, node_v_layout)
        var aux_view = TileTensor(aux.buf, aux_layout)

        comptime kernel = _vlasov_rhs_kernel_1d1v[
            Self.P_x, Self.P_v,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
            type_of(node_v_layout), type_of(aux_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            node_v_view, aux_view, Int64(mesh.num_v),
            self.app.charge_mass_ratio / self.app.skin_depth_norm,
            Real(1.0) / mesh.jacobian_x, Real(1.0) / mesh.jacobian_v, Int64(mesh.num_phase_elements),
            grid_dim=(mesh.num_v, mesh.num_x),
            block_dim=(P1x, P1v),
        )
