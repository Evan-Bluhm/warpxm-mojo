# 3D nodal DG spatial solver for 5-moment Euler.
#
# Tensor-product extension of `dg_euler_2d.mojo` with `NC = EULER_NC = 5`
# and the 3D Euler flux helpers. Six face corrections per element instead
# of four. Today: periodic boundaries only.

from std.collections import InlineArray
from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, Idx, row_major, stack_allocation

from warpxm.apps.euler_3d import Euler3D
from warpxm.apps.euler_2d import EULER_NC, IDX_MX, IDX_MY, IDX_MZ
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_3d import (
    BlockMesh3D, GHOST_NEG_3D, GHOST_POS_3D,
    FACE_DIR_LEFT_3D, FACE_DIR_RIGHT_3D,
    FACE_DIR_BOTTOM_3D, FACE_DIR_TOP_3D,
    FACE_DIR_DOWN_3D, FACE_DIR_UP_3D,
)
from warpxm.variable.state_3d import DGState3D


@always_inline
def _apply_bc_euler_3d(
    q_interior: InlineArray[Real, EULER_NC],
    bc_kind: Int,
    face_axis: Int,
) -> InlineArray[Real, EULER_NC]:
    """Ghost-state synthesis for a 3D Euler boundary face.

    `face_axis` ∈ {0, 1, 2} selects which momentum component flips for
    `BC_WALL_FREESLIP` (the normal one to the wall). `BC_COPYOUT` and any
    other kind fall back to copy-out (ghost = interior).
    """
    var q: InlineArray[Real, EULER_NC] = q_interior
    if bc_kind == BC_WALL_FREESLIP:
        if face_axis == 0:
            q[IDX_MX] = -q_interior[IDX_MX]
        elif face_axis == 1:
            q[IDX_MY] = -q_interior[IDX_MY]
        else:
            q[IDX_MZ] = -q_interior[IDX_MZ]
    return q


@always_inline
def _load_node[
    P: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    elem: Int,
    local_node: Int,
) -> InlineArray[Real, EULER_NC]:
    comptime assert q.flat_rank == 1
    comptime num_nodes_per_elem = (P + 1) ** 3
    var base = (elem * num_nodes_per_elem + local_node) * EULER_NC
    var out: InlineArray[Real, EULER_NC] = InlineArray[Real, EULER_NC](fill=Real(0.0))
    comptime for c in range(EULER_NC):
        out[c] = rebind[Scalar[WXM_DTYPE]](q[base + c])
    return out


def _euler_rhs_kernel_3d[
    P: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_bc: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    gamma: Real, rho_min: Real, press_min: Real,
    jacobian_x: Real, jacobian_y: Real, jacobian_z: Real,
    num_elements: Int,
):
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1 * P1
    comptime NC = EULER_NC
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1
    comptime assert face_bc.flat_rank == 1

    var e = Int(block_idx.x)
    var in_ = Int(thread_idx.x)
    var jn = Int(thread_idx.y)
    var kn = Int(thread_idx.z)
    if e >= num_elements or in_ >= P1 or jn >= P1 or kn >= P1:
        return

    var local_n = kn * P1 * P1 + jn * P1 + in_

    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[P1 * P1 * P1, NC, 3]())

    var q_n = _load_node[P](q, e, local_n)
    var fx_n = Euler3D.internal_flux_x_static(q_n, gamma, rho_min, press_min)
    var fy_n = Euler3D.internal_flux_y_static(q_n, gamma, rho_min, press_min)
    var fz_n = Euler3D.internal_flux_z_static(q_n, gamma, rho_min, press_min)
    comptime for c in range(NC):
        f_shared[local_n, c, 0] = rebind[f_shared.ElementType](fx_n[c])
        f_shared[local_n, c, 1] = rebind[f_shared.ElementType](fy_n[c])
        f_shared[local_n, c, 2] = rebind[f_shared.ElementType](fz_n[c])

    barrier()

    var d_flat = NodalBasis1D[P].diff_matrix_flat()
    var dfdx: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    var dgdy: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))
    var dhdz: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    comptime for kx in range(P1):
        var d_in_kx: Scalar[WXM_DTYPE] = d_flat[in_ * P1 + kx]
        var k_node = kn * P1 * P1 + jn * P1 + kx
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 0])
            dfdx[c] += d_in_kx * v
    comptime for ky in range(P1):
        var d_jn_ky: Scalar[WXM_DTYPE] = d_flat[jn * P1 + ky]
        var k_node = kn * P1 * P1 + ky * P1 + in_
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 1])
            dgdy[c] += d_jn_ky * v
    comptime for kz in range(P1):
        var d_kn_kz: Scalar[WXM_DTYPE] = d_flat[kn * P1 + kz]
        var k_node = kz * P1 * P1 + jn * P1 + in_
        comptime for c in range(NC):
            var v = rebind[Scalar[WXM_DTYPE]](f_shared[k_node, c, 2])
            dhdz[c] += d_kn_kz * v

    var weights = NodalBasis1D[P].weights()
    var w_0: Scalar[WXM_DTYPE] = weights[0]
    var w_P: Scalar[WXM_DTYPE] = weights[P]
    var surface: InlineArray[Real, NC] = InlineArray[Real, NC](fill=Real(0.0))

    if in_ == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_LEFT_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = _apply_bc_euler_3d(q_n, bc_kind, face_axis=0)
        else:
            q_nbr = _load_node[P](q, neg_e, kn * P1 * P1 + jn * P1 + P)
        var f_star = Euler3D.numerical_flux_static(
            q_nbr, q_n, Real(1.0), Real(0.0), Real(0.0), gamma, rho_min, press_min
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fx_n[c]) / (jacobian_x * w_0)
    if in_ == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_RIGHT_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = _apply_bc_euler_3d(q_n, bc_kind, face_axis=0)
        else:
            q_nbr = _load_node[P](q, pos_e, kn * P1 * P1 + jn * P1 + 0)
        var f_star = Euler3D.numerical_flux_static(
            q_n, q_nbr, Real(1.0), Real(0.0), Real(0.0), gamma, rho_min, press_min
        )
        comptime for c in range(NC):
            surface[c] += (fx_n[c] - f_star[c]) / (jacobian_x * w_P)
    if jn == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_BOTTOM_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = _apply_bc_euler_3d(q_n, bc_kind, face_axis=1)
        else:
            q_nbr = _load_node[P](q, neg_e, kn * P1 * P1 + P * P1 + in_)
        var f_star = Euler3D.numerical_flux_static(
            q_nbr, q_n, Real(0.0), Real(1.0), Real(0.0), gamma, rho_min, press_min
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fy_n[c]) / (jacobian_y * w_0)
    if jn == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_TOP_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = _apply_bc_euler_3d(q_n, bc_kind, face_axis=1)
        else:
            q_nbr = _load_node[P](q, pos_e, kn * P1 * P1 + 0 * P1 + in_)
        var f_star = Euler3D.numerical_flux_static(
            q_n, q_nbr, Real(0.0), Real(1.0), Real(0.0), gamma, rho_min, press_min
        )
        comptime for c in range(NC):
            surface[c] += (fy_n[c] - f_star[c]) / (jacobian_y * w_P)
    if kn == 0:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_DOWN_3D]))
        var neg_e = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if neg_e == Int(GHOST_NEG_3D):
            q_nbr = _apply_bc_euler_3d(q_n, bc_kind, face_axis=2)
        else:
            q_nbr = _load_node[P](q, neg_e, P * P1 * P1 + jn * P1 + in_)
        var f_star = Euler3D.numerical_flux_static(
            q_nbr, q_n, Real(0.0), Real(0.0), Real(1.0), gamma, rho_min, press_min
        )
        comptime for c in range(NC):
            surface[c] += (f_star[c] - fz_n[c]) / (jacobian_z * w_0)
    if kn == P:
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 6 + FACE_DIR_UP_3D]))
        var pos_e = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        var bc_kind = Int(rebind[Scalar[INDEX_DTYPE]](face_bc[f_id]))
        var q_nbr: InlineArray[Real, NC]
        if pos_e == Int(GHOST_POS_3D):
            q_nbr = _apply_bc_euler_3d(q_n, bc_kind, face_axis=2)
        else:
            q_nbr = _load_node[P](q, pos_e, 0 * P1 * P1 + jn * P1 + in_)
        var f_star = Euler3D.numerical_flux_static(
            q_n, q_nbr, Real(0.0), Real(0.0), Real(1.0), gamma, rho_min, press_min
        )
        comptime for c in range(NC):
            surface[c] += (fz_n[c] - f_star[c]) / (jacobian_z * w_P)

    var base = (e * nodes_per_elem + local_n) * NC
    comptime for c in range(NC):
        var v: Scalar[WXM_DTYPE] = (
            -dfdx[c] / jacobian_x
            - dgdy[c] / jacobian_y
            - dhdz[c] / jacobian_z
            + surface[c]
        )
        rhs[base + c] = rebind[rhs.ElementType](v)


struct DGEulerSolver3D[P: Int]:
    """Host-side launcher for the 3D 5-moment Euler DG kernel."""

    var app: Euler3D

    def __init__(out self, app: Euler3D):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, EULER_NC],
        mut rhs: DGState3D[Self.P, EULER_NC],
    ) raises:
        comptime P1 = Self.P + 1

        var q_layout = row_major(Idx(q.buf_len))
        var face_layout = row_major(Idx(mesh.num_faces))
        var ef_layout = row_major(Idx(mesh.num_elements * 6))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var fbc_view = TileTensor(mesh.face_bc_buf, face_layout)

        comptime kernel = _euler_rhs_kernel_3d[
            Self.P, type_of(q_layout), type_of(q_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, fn_view, fp_view, ef_view, fbc_view,
            self.app.gas_gamma, self.app.min_density, self.app.min_pressure,
            mesh.jacobian_x, mesh.jacobian_y, mesh.jacobian_z,
            mesh.num_elements,
            grid_dim=mesh.num_elements,
            block_dim=(P1, P1, P1),
        )
