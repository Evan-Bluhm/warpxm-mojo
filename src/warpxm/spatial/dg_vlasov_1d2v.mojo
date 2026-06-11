# 1D2V nodal DG spatial solver for the kinetic Vlasov-Maxwell equation.
#
# Extension of `dg_vlasov_1d1v.mojo` to one physical-space and two
# velocity-space dimensions. The unknown is `f(x, v1, v2, t)`; the kernel
# evaluates the DG residual
#
#     ∂f/∂t = -∂(v1 · f)/∂x
#             - ∂(a_v1 · f)/∂v1
#             - ∂(a_v2 · f)/∂v2
#             + surface contributions
#
# at every tensor-product GLL phase node. The wave speed along the x-axis
# is the local `v1` coordinate of the current node; the wave speeds along
# the two velocity axes are the per-node Lorentz accelerations
#
#     a_v1 = (q/m)/λ · (E_x + v2 · B_z)
#     a_v2 = (q/m)/λ · (E_y − v1 · B_z)
#
# read from the Maxwell aux state. Both `v × B` cross terms vanished in
# 1D1V (no transverse v-components); here they do real work — they are
# what gyrates a particle around `B_z`.
#
# Phase mesh layout (matches `mesh/phase_mesh_1d2v.mojo`):
#   axis 0 = x  (may be periodic; otherwise caller-supplied bc_left/right)
#   axis 1 = v1 (= v_x; truncated outflow at ±v1_max via `BC_VLASOV_OUTFLOW`)
#   axis 2 = v2 (= v_y; truncated outflow at ±v2_max via `BC_VLASOV_OUTFLOW`)
#
# Local node convention (matches `variable/phase_state_1d2v.mojo`):
#     local_n = (iv2 · (P_v1 + 1) + iv1) · (P_x + 1) + ix
# i.e. v2 outer / v1 middle / x inner — same shape as
# `spatial/dg_core_3d.mojo`'s `kn · P1² + jn · P1 + in_` so the kernel
# structure carries over almost verbatim.

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, Idx, row_major, stack_allocation

from warpxm.apps.vlasovmaxwell import (
    VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC,
    AUX_EX, AUX_EY, AUX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_VLASOV_OUTFLOW
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import (
    GHOST_NEG_3D, GHOST_POS_3D,
    FACE_DIR_LEFT_3D, FACE_DIR_RIGHT_3D,
    FACE_DIR_BOTTOM_3D, FACE_DIR_TOP_3D,
    FACE_DIR_DOWN_3D, FACE_DIR_UP_3D,
)
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state_1d import DGState1D


@always_inline
def _load_phase_node_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    phase_e: Int,
    local_node: Int,
) -> InlineArray[Real, NC]:
    comptime assert q.flat_rank == 1
    comptime nodes_per_phase = (P_x + 1) * (P_v1 + 1) * (P_v2 + 1)
    var base = (phase_e * nodes_per_phase + local_node) * NC
    var out: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    comptime for c in range(NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out


def _vlasov_rhs_kernel_1d2v[
    P_x: Int, P_v1: Int, P_v2: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
    NodeV1Layout: TensorLayout, NodeV2Layout: TensorLayout,
    AuxLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    node_v1: TileTensor[WXM_DTYPE, NodeV1Layout, MutAnyOrigin],
    node_v2: TileTensor[WXM_DTYPE, NodeV2Layout, MutAnyOrigin],
    aux: TileTensor[WXM_DTYPE, AuxLayout, MutAnyOrigin],
    num_v1: Int, num_v2: Int,
    charge_mass_ratio: Real, skin_depth_norm: Real,
    jacobian_x: Real, jacobian_v1: Real, jacobian_v2: Real,
    num_phase_elements: Int,
):
    """Per-phase-node DG residual for 1D2V Vlasov.

    The two Lorentz accelerations are formed per-node from the local
    `(v1, v2)` coordinates and the Maxwell aux state at the current
    physical x-node (`E_x`, `E_y`, `B_z`). Both sides of any axis-aligned
    face share the same wave speed (aux depends only on `x`; `v_i` is
    constant along any face perpendicular to the i-axis).

    Thread layout: `block_idx.x = phase_e`,
    `(thread_idx.x, thread_idx.y, thread_idx.z) = (ix, iv1, iv2)`.
    """
    comptime P1x = P_x + 1
    comptime P1v1 = P_v1 + 1
    comptime P1v2 = P_v2 + 1
    comptime nodes_per_phase = P1x * P1v1 * P1v2
    comptime NC = VLASOV_NC
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1
    comptime assert face_bc.flat_rank == 1
    comptime assert node_v1.flat_rank == 1
    comptime assert node_v2.flat_rank == 1
    comptime assert aux.flat_rank == 1

    var phase_e = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    var iv1 = Int(thread_idx.y)
    var iv2 = Int(thread_idx.z)
    if (
        phase_e >= num_phase_elements
        or ix >= P1x or iv1 >= P1v1 or iv2 >= P1v2
    ):
        return

    # Decompose phase_e back into (ex, ev1, ev2). PhaseState1D2V layout:
    #     phase_e = (ex · num_v1 + ev1) · num_v2 + ev2.
    var ev2 = phase_e % num_v2
    var ex_ev1 = phase_e // num_v2
    var ev1 = ex_ev1 % num_v1
    var ex = ex_ev1 // num_v1
    var local_n = (iv2 * P1v1 + iv1) * P1x + ix

    # Per-node v1, v2 coordinates. Both buffers store one entry per
    # (e, i) pair on their own axis.
    var v1_n: Real = rebind[Scalar[WXM_DTYPE]](node_v1[ev1 * P1v1 + iv1])
    var v2_n: Real = rebind[Scalar[WXM_DTYPE]](node_v2[ev2 * P1v2 + iv2])

    # Maxwell aux at the current x-node (per-x-node flat storage; the
    # aux is independent of v). For 1D2V the only field components that
    # enter `a_v1`, `a_v2` are `E_x`, `E_y`, `B_z`.
    var node_x = ex * P1x + ix
    var ex_field: Real = rebind[Scalar[WXM_DTYPE]](
        aux[node_x * VLASOV_AUX_NC + AUX_EX]
    )
    var ey_field: Real = rebind[Scalar[WXM_DTYPE]](
        aux[node_x * VLASOV_AUX_NC + AUX_EY]
    )
    var bz_field: Real = rebind[Scalar[WXM_DTYPE]](
        aux[node_x * VLASOV_AUX_NC + AUX_BZ]
    )
    var av1_n: Real = VlasovMaxwell.acceleration_v1_1d2v_static(
        ex_field, bz_field, v2_n, charge_mass_ratio, skin_depth_norm,
    )
    var av2_n: Real = VlasovMaxwell.acceleration_v2_1d2v_static(
        ey_field, bz_field, v1_n, charge_mass_ratio, skin_depth_norm,
    )

    # Shared-memory cache of the three per-node fluxes for this phase
    # cell, so the three volume sweeps can re-read neighbour-node fluxes
    # without going back to global memory.
    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[P1x * P1v1 * P1v2, NC, 3]())

    var f_n = _load_phase_node_1d2v[P_x, P_v1, P_v2, NC](q, phase_e, local_n)
    var fx_n = VlasovMaxwell.internal_flux_x_static[NC](f_n, v1_n)
    var fv1_n = VlasovMaxwell.internal_flux_v1_static[NC](f_n, av1_n)
    var fv2_n = VlasovMaxwell.internal_flux_v2_static[NC](f_n, av2_n)
    comptime for c in range(NC):
        f_shared[local_n, c, 0] = rebind[f_shared.ElementType](fx_n[c])
        f_shared[local_n, c, 1] = rebind[f_shared.ElementType](fv1_n[c])
        f_shared[local_n, c, 2] = rebind[f_shared.ElementType](fv2_n[c])

    barrier()

    # --- Volume term ------------------------------------------------------
    # Three tensor-product differentiation sweeps, one per axis. Each
    # axis uses its own GLL diff matrix (P_x, P_v1, P_v2 may differ).
    var dx_flat = NodalBasis1D[P_x].diff_matrix_flat()
    var dv1_flat = NodalBasis1D[P_v1].diff_matrix_flat()
    var dv2_flat = NodalBasis1D[P_v2].diff_matrix_flat()
    var dfdx: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    var dgdv1: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    var dhdv2: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    comptime for kx in range(P1x):
        var dx_ix_kx: Scalar[WXM_DTYPE] = dx_flat[ix * P1x + kx]
        var k_node = (iv2 * P1v1 + iv1) * P1x + kx
        comptime for c in range(NC):
            var fx_kx = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 0])
            dfdx[c] += dx_ix_kx * fx_kx

    comptime for ky in range(P1v1):
        var dv1_iv1_ky: Scalar[WXM_DTYPE] = dv1_flat[iv1 * P1v1 + ky]
        var k_node = (iv2 * P1v1 + ky) * P1x + ix
        comptime for c in range(NC):
            var fv1_ky = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 1])
            dgdv1[c] += dv1_iv1_ky * fv1_ky

    comptime for kz in range(P1v2):
        var dv2_iv2_kz: Scalar[WXM_DTYPE] = dv2_flat[iv2 * P1v2 + kz]
        var k_node = (kz * P1v1 + iv1) * P1x + ix
        comptime for c in range(NC):
            var fv2_kz = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 2])
            dhdv2[c] += dv2_iv2_kz * fv2_kz

    # --- Surface term -----------------------------------------------------
    # Six boundary-node branches. Wave speeds: v1 on x-faces, av1 on
    # v1-faces, av2 on v2-faces — all single-valued across the face since
    # they depend only on coordinates / aux that are themselves constant
    # along the face normal.
    var weights_x = NodalBasis1D[P_x].weights()
    var weights_v1 = NodalBasis1D[P_v1].weights()
    var weights_v2 = NodalBasis1D[P_v2].weights()
    var wx_0: Scalar[WXM_DTYPE] = weights_x[0]
    var wx_P: Scalar[WXM_DTYPE] = weights_x[P_x]
    var wv1_0: Scalar[WXM_DTYPE] = weights_v1[0]
    var wv1_P: Scalar[WXM_DTYPE] = weights_v1[P_v1]
    var wv2_0: Scalar[WXM_DTYPE] = weights_v2[0]
    var wv2_P: Scalar[WXM_DTYPE] = weights_v2[P_v2]

    var surface: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    # x-LEFT (axis-0 negative).
    if ix == 0:
        var fL = Int(rebind[Scalar[INDEX_DTYPE]](
            elem_face_dir[phase_e * 6 + FACE_DIR_LEFT_3D]
        ))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fL]))
        var q_neg: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_neg = f_n  # BC_COPYOUT fallback on non-periodic x.
        else:
            q_neg = _load_phase_node_1d2v[P_x, P_v1, P_v2, NC](
                q, neg_e, (iv2 * P1v1 + iv1) * P1x + P_x,
            )
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            q_neg, f_n, v1_n,
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fx_n[c]) / (jacobian_x * wx_0)

    # x-RIGHT (axis-0 positive).
    if ix == P_x:
        var fR = Int(rebind[Scalar[INDEX_DTYPE]](
            elem_face_dir[phase_e * 6 + FACE_DIR_RIGHT_3D]
        ))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fR]))
        var q_pos: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_pos = f_n
        else:
            q_pos = _load_phase_node_1d2v[P_x, P_v1, P_v2, NC](
                q, pos_e, (iv2 * P1v1 + iv1) * P1x + 0,
            )
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            f_n, q_pos, v1_n,
        )
        comptime for c in range(NC):
            surface[c] += (fx_n[c] - f_star[c]) / (jacobian_x * wx_P)

    # v1-BOTTOM (axis-1 negative).
    if iv1 == 0:
        var fB = Int(rebind[Scalar[INDEX_DTYPE]](
            elem_face_dir[phase_e * 6 + FACE_DIR_BOTTOM_3D]
        ))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fB]))
        var q_neg: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            var bc = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fB]))
            if bc == BC_VLASOV_OUTFLOW:
                q_neg = InlineArray[Real, NC](fill=Real(0.0))
            else:
                q_neg = f_n
        else:
            q_neg = _load_phase_node_1d2v[P_x, P_v1, P_v2, NC](
                q, neg_e, (iv2 * P1v1 + P_v1) * P1x + ix,
            )
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            q_neg, f_n, av1_n,
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fv1_n[c]) / (jacobian_v1 * wv1_0)

    # v1-TOP (axis-1 positive).
    if iv1 == P_v1:
        var fT = Int(rebind[Scalar[INDEX_DTYPE]](
            elem_face_dir[phase_e * 6 + FACE_DIR_TOP_3D]
        ))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fT]))
        var q_pos: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            var bc = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fT]))
            if bc == BC_VLASOV_OUTFLOW:
                q_pos = InlineArray[Real, NC](fill=Real(0.0))
            else:
                q_pos = f_n
        else:
            q_pos = _load_phase_node_1d2v[P_x, P_v1, P_v2, NC](
                q, pos_e, (iv2 * P1v1 + 0) * P1x + ix,
            )
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            f_n, q_pos, av1_n,
        )
        comptime for c in range(NC):
            surface[c] += (fv1_n[c] - f_star[c]) / (jacobian_v1 * wv1_P)

    # v2-DOWN (axis-2 negative).
    if iv2 == 0:
        var fD = Int(rebind[Scalar[INDEX_DTYPE]](
            elem_face_dir[phase_e * 6 + FACE_DIR_DOWN_3D]
        ))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fD]))
        var q_neg: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            var bc = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fD]))
            if bc == BC_VLASOV_OUTFLOW:
                q_neg = InlineArray[Real, NC](fill=Real(0.0))
            else:
                q_neg = f_n
        else:
            q_neg = _load_phase_node_1d2v[P_x, P_v1, P_v2, NC](
                q, neg_e, (P_v2 * P1v1 + iv1) * P1x + ix,
            )
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            q_neg, f_n, av2_n,
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fv2_n[c]) / (jacobian_v2 * wv2_0)

    # v2-UP (axis-2 positive).
    if iv2 == P_v2:
        var fU = Int(rebind[Scalar[INDEX_DTYPE]](
            elem_face_dir[phase_e * 6 + FACE_DIR_UP_3D]
        ))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fU]))
        var q_pos: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            var bc = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[fU]))
            if bc == BC_VLASOV_OUTFLOW:
                q_pos = InlineArray[Real, NC](fill=Real(0.0))
            else:
                q_pos = f_n
        else:
            q_pos = _load_phase_node_1d2v[P_x, P_v1, P_v2, NC](
                q, pos_e, (0 * P1v1 + iv1) * P1x + ix,
            )
        var f_star = VlasovMaxwell.numerical_flux_static[NC](
            f_n, q_pos, av2_n,
        )
        comptime for c in range(NC):
            surface[c] += (fv2_n[c] - f_star[c]) / (jacobian_v2 * wv2_P)

    # --- Write back -------------------------------------------------------
    var base = (phase_e * nodes_per_phase + local_n) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = (
            -dfdx[c] / jacobian_x
            - dgdv1[c] / jacobian_v1
            - dhdv2[c] / jacobian_v2
            + surface[c]
        )
        rhs[base + c] = rebind[rhs.ElementType](v)


struct DGVlasovSolver1D2V[P_x: Int, P_v1: Int, P_v2: Int]:
    """Host-side launcher for the 1D2V Vlasov DG kernel.

    Reads `E_x`, `E_y`, `B_z` from a per-x-node Maxwell aux buffer
    (`DGState1D[P_x, VLASOV_AUX_NC]`) to form the two Lorentz
    accelerations. As in the 1D1V launcher, the aux buffer is currently
    user-populated; a coupled Maxwell-Vlasov stepper arrives in slice 21."""

    var app: VlasovMaxwell

    def __init__(out self, app: VlasovMaxwell):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D2V[Self.P_x, Self.P_v1, Self.P_v2],
        mut q: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC],
        mut aux: DGState1D[Self.P_x, VLASOV_AUX_NC],
        mut rhs: PhaseState1D2V[Self.P_x, Self.P_v1, Self.P_v2, VLASOV_NC],
    ) raises:
        comptime P1x = Self.P_x + 1
        comptime P1v1 = Self.P_v1 + 1
        comptime P1v2 = Self.P_v2 + 1

        var q_layout = row_major(Idx(q.buf_len))
        var face_layout = row_major(Idx(mesh.num_faces))
        var ef_layout = row_major(Idx(mesh.num_phase_elements * 6))
        var node_v1_layout = row_major(Idx(mesh.num_v1 * P1v1))
        var node_v2_layout = row_major(Idx(mesh.num_v2 * P1v2))
        var aux_layout = row_major(Idx(aux.buf_len))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)
        var node_v1_view = TileTensor(mesh.node_v1_buf, node_v1_layout)
        var node_v2_view = TileTensor(mesh.node_v2_buf, node_v2_layout)
        var aux_view = TileTensor(aux.buf, aux_layout)

        comptime kernel = _vlasov_rhs_kernel_1d2v[
            Self.P_x, Self.P_v1, Self.P_v2,
            type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
            type_of(node_v1_layout), type_of(node_v2_layout),
            type_of(aux_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            node_v1_view, node_v2_view, aux_view,
            mesh.num_v1, mesh.num_v2,
            self.app.charge_mass_ratio, self.app.skin_depth_norm,
            mesh.jacobian_x, mesh.jacobian_v1, mesh.jacobian_v2,
            mesh.num_phase_elements,
            grid_dim=mesh.num_phase_elements,
            block_dim=(P1x, P1v1, P1v2),
        )
