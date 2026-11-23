# Nodal DG spatial solver for the kinetic Vlasov equation, generic over the
# phase-space dimensions.
#
# The unknown is `f(x, v, t)` with `XDIM` configuration axes and `VDIM`
# velocity axes (1D1V, 1D2V, 1D3V, 2D2V, 2D3V, 3D3V). The kernel evaluates
# the DG residual of
#
#     ∂f/∂t = − Σ_a ∂(v_a f)/∂x_a − Σ_d ∂(a_d f)/∂v_d
#
# at every tensor-product GLL phase node, with the Lorentz acceleration
#
#     a = (q/m)/λ · (E + v × B)
#
# formed from the Maxwell state at the node's configuration-space point.
# All six field components enter in 3V:
#
#     a_1 = (q/m)/λ · (E_x + v_2 B_z − v_3 B_y)
#     a_2 = (q/m)/λ · (E_y + v_3 B_x − v_1 B_z)
#     a_3 = (q/m)/λ · (E_z + v_1 B_y − v_2 B_x)
#
# and the terms of absent velocity components drop out at compile time
# (`VlasovMaxwell.lorentz_acceleration_static`).
#
# Two properties of the uniform Cartesian phase mesh shape the kernel:
#
#   1. The advection speed along an axis does not depend on the coordinate
#      of that axis: `v_a` is independent of `x_a`, and `a_d` is
#      independent of `v_d`. So the speed is constant along a line of
#      nodes and single-valued on a face, the volume term is
#      `c · Σ_m D[i, m] f_m`, and the upwind surface term reduces to
#      `max(c, 0) · (f_nbr − f)` on the negative face and
#      `min(c, 0) · (f − f_nbr)` on the positive face. Only `f` itself is
#      cached in shared memory, not one flux per axis.
#   2. Neighbours are found by index arithmetic (`mesh/phase_mesh.mojo`),
#      so there are no face tables to load.
#
# Configuration boundaries are periodic, copy-out, zero-inflow, or a
# specular wall (`BC_VLASOV_REFLECT`), whose ghost state is the interior
# state at the mirrored normal velocity. With the upwind flux the mass
# flux through such a wall cancels node by node between `v_n` and `−v_n`.
#
# Thread layout: one block per phase element, `thread_idx.x` the
# configuration node and `thread_idx.y` the velocity node. The launch grid
# is `(lead, n_{ND−2}, n_{ND−1})`, `lead` being the flattened index of the
# leading `ND − 2` axes, so the element indices of the last two axes come
# straight from the block index and only the leading ones are unpacked,
# with 32-bit divisions (none up to three axes, one for 1D3V and 2D2V).

from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major, stack_allocation

from warpxm.apps.vlasovmaxwell import (
    VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC,
    AUX_EX, AUX_EY, AUX_EZ, AUX_BX, AUX_BY, AUX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import (
    BC_INTERIOR, BC_COPYOUT, BC_VLASOV_REFLECT, BC_HALO,
)
from warpxm.core.math_compat import fabs, fmax, fmin
from warpxm.core.types import Real, WXM_DTYPE, WXM_TINY, WXM_INF
from warpxm.mesh.phase_mesh import PhaseMesh, PhaseInts, PhaseReals
from warpxm.parallel.halo import exchange_halo
from warpxm.parallel.mpi import USE_MPI, global_max
from warpxm.spatial.dg_launch import RealTensor
from warpxm.variable.phase_state import PhaseState
from warpxm.variable.state import DGState


@always_inline
def _face_slot[ND: Int, axis: Int](n: Array[Int, ND], e: Array[Int, ND]) -> Int:
    """Position of element `e` in the layer of elements on a face normal
    to `axis`: its index with that axis left out, which is how the ghosts
    of a partition face are ordered (`PhaseMesh`)."""
    var slot = 0
    comptime for b in range(ND):
        comptime if b != axis:
            slot = slot * n[b] + e[b]
    return slot


def _vlasov_rhs_kernel[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int](
    q: RealTensor,
    rhs: RealTensor,
    node_v: RealTensor,
    aux: RealTensor,
    counts: PhaseInts,
    x_bc: PhaseInts,
    ghost_first: PhaseInts,
    inv_jacobian: PhaseReals,
    accel_factor: Real,
):
    """Per-phase-node DG residual of the Vlasov equation.

    `counts[a]` is the element count of axis `a`, `x_bc[2·a + side]` the BC
    kind of a configuration boundary (`BC_INTERIOR` for a periodic axis,
    `BC_HALO` for a face shared with another rank, whose ghost elements
    start at `ghost_first[2·a + side]`),
    `inv_jacobian[a]` is `2 / Δ_a` and `accel_factor` is `(q/m) / λ`.

    Every block is exactly one phase element and the grid is exactly the
    mesh, so no thread is out of range and all of them reach the barrier.
    """
    comptime ND = XDIM + VDIM
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime NXN = P1x ** XDIM
    comptime NVN = P1v ** VDIM
    comptime NPE = NXN * NVN
    comptime assert VLASOV_NC == 1, "the kernel advances the scalar f"

    var n = Array[Int, ND](fill=1)
    comptime for a in range(ND):
        n[a] = Int(counts[a])

    # --- Element index along every axis -----------------------------------
    var e = Array[Int, ND](fill=0)
    comptime if ND == 2:
        e[0] = Int(block_idx.x)
        e[1] = Int(block_idx.y)
    else:
        e[ND - 1] = Int(block_idx.z)
        e[ND - 2] = Int(block_idx.y)
        var rem = UInt32(block_idx.x)
        comptime for k in range(ND - 3):
            comptime a = ND - 3 - k
            var n_a = UInt32(n[a])
            var quo = rem // n_a
            e[a] = Int(rem - quo * n_a)
            rem = quo
        e[0] = Int(rem)

    # Element stride of each axis (last axis innermost) and the flat index.
    var stride = Array[Int, ND](fill=1)
    comptime for k in range(ND - 1):
        comptime a = ND - 2 - k
        stride[a] = stride[a + 1] * n[a + 1]
    var phase_e = 0
    comptime for a in range(ND):
        phase_e += e[a] * stride[a]

    # --- Node index along every axis (first axis innermost) ---------------
    var xn = Int(thread_idx.x)
    var vn = Int(thread_idx.y)
    var local_n = vn * NXN + xn
    var i = Array[Int, ND](fill=0)
    comptime for a in range(ND):
        comptime if a < XDIM:
            i[a] = (xn // (P1x ** a)) % P1x
        else:
            i[a] = (vn // (P1v ** (a - XDIM))) % P1v

    # --- Velocity coordinates of this node --------------------------------
    var v = Array[Real, 3](fill=Real(0.0))
    var v_offset = 0
    comptime for d in range(VDIM):
        v[d] = rebind[Real](
            node_v[v_offset + e[XDIM + d] * P1v + i[XDIM + d]]
        )
        v_offset += n[XDIM + d] * P1v

    # --- Fields at this node's configuration point ------------------------
    # The companion `BlockMesh` numbers its elements with x innermost.
    var xe_block = 0
    var x_stride = 1
    comptime for a in range(XDIM):
        xe_block += e[a] * x_stride
        x_stride *= n[a]
    var aux_base = (xe_block * NXN + xn) * VLASOV_AUX_NC
    var ex_f = rebind[Real](aux[aux_base + AUX_EX])
    var ey_f = rebind[Real](aux[aux_base + AUX_EY])
    var ez_f = rebind[Real](aux[aux_base + AUX_EZ])
    var bx_f = rebind[Real](aux[aux_base + AUX_BX])
    var by_f = rebind[Real](aux[aux_base + AUX_BY])
    var bz_f = rebind[Real](aux[aux_base + AUX_BZ])

    # Advection speed along each axis, already divided by the Jacobian.
    var speed = Array[Real, ND](fill=Real(0.0))
    comptime for a in range(ND):
        comptime if a < XDIM:
            speed[a] = v[a] * inv_jacobian[a]
        else:
            speed[a] = VlasovMaxwell.lorentz_acceleration_static[
                VDIM, a - XDIM,
            ](
                ex_f, ey_f, ez_f, bx_f, by_f, bz_f,
                v[0], v[1], v[2], accel_factor,
            ) * inv_jacobian[a]

    # --- Shared-memory copy of this element's f ---------------------------
    var f_shared = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[NPE]())
    var node_base = phase_e * NPE
    var f_n = rebind[Real](q[node_base + local_n])
    f_shared[local_n] = rebind[f_shared.ElementType](f_n)

    barrier()

    # FP64 and local-memory hygiene (see CLAUDE.md, "Kernel performance
    # rules"): reciprocal Jacobians come from the launcher, the weights
    # are comptime reciprocals, the differentiation matrix is read through
    # `diff_entry`, and every scratch array is indexed at compile time.
    var out = Real(0.0)
    comptime for a in range(ND):
        comptime P_a = P_x if a < XDIM else P_v
        comptime node_stride = (
            (P1x ** a) if a < XDIM else NXN * (P1v ** (a - XDIM))
        )
        comptime inv_w0: Real = Real(1.0) / NodalBasis1D[P_a].weights()[0]
        comptime inv_wP: Real = Real(1.0) / NodalBasis1D[P_a].weights()[P_a]
        var along = i[a]
        var c = speed[a]

        # Volume term: c · Σ_m D[along, m] f_m along this axis's node line.
        var line_start = local_n - along * node_stride
        var d_sum = Real(0.0)
        comptime for m in range(P_a + 1):
            var d: Real = NodalBasis1D[P_a].diff_entry[m](along)
            d_sum += d * rebind[Real](f_shared[line_start + m * node_stride])
        out -= c * d_sum

        # Specular wall on a configuration axis: the ghost state is this
        # same wall node at the mirrored normal velocity, v_a → −v_a. The
        # velocity mesh is symmetric about zero there (checked by
        # `PhaseMesh`), so that is velocity element `n − 1 − e`, node
        # `P_v − i` along velocity axis a: an index offset, no
        # interpolation.
        var mirror_elem = 0
        var mirror_node = 0
        comptime if a < XDIM:
            mirror_elem = (
                (n[XDIM + a] - 1 - 2 * e[XDIM + a]) * stride[XDIM + a]
            )
            mirror_node = (P_v - 2 * i[XDIM + a]) * (NXN * (P1v ** a))

        # Surface term, negative face: upwind inflow when c > 0.
        if along == 0:
            var nbr_node = local_n + P_a * node_stride
            var f_nbr = Real(0.0)
            if e[a] > 0:
                f_nbr = rebind[Real](
                    q[(phase_e - stride[a]) * NPE + nbr_node]
                )
            else:
                # Velocity boundaries are outflow: the ghost state is 0.
                comptime if a < XDIM:
                    var kind = Int(x_bc[2 * a])
                    if kind == BC_INTERIOR:
                        f_nbr = rebind[Real](q[
                            (phase_e + (n[a] - 1) * stride[a]) * NPE
                            + nbr_node
                        ])
                    elif kind == BC_HALO:
                        f_nbr = rebind[Real](q[
                            (
                                Int(ghost_first[2 * a])
                                + _face_slot[ND, a](n, e)
                            ) * NPE
                            + nbr_node
                        ])
                    elif kind == BC_COPYOUT:
                        f_nbr = f_n
                    elif kind == BC_VLASOV_REFLECT:
                        f_nbr = rebind[Real](q[
                            (phase_e + mirror_elem) * NPE
                            + local_n + mirror_node
                        ])
            out += fmax(c, Real(0.0)) * inv_w0 * (f_nbr - f_n)

        # Surface term, positive face: upwind inflow when c < 0.
        if along == P_a:
            var nbr_node = local_n - P_a * node_stride
            var f_nbr = Real(0.0)
            if e[a] < n[a] - 1:
                f_nbr = rebind[Real](
                    q[(phase_e + stride[a]) * NPE + nbr_node]
                )
            else:
                comptime if a < XDIM:
                    var kind = Int(x_bc[2 * a + 1])
                    if kind == BC_INTERIOR:
                        f_nbr = rebind[Real](q[
                            (phase_e - (n[a] - 1) * stride[a]) * NPE
                            + nbr_node
                        ])
                    elif kind == BC_HALO:
                        f_nbr = rebind[Real](q[
                            (
                                Int(ghost_first[2 * a + 1])
                                + _face_slot[ND, a](n, e)
                            ) * NPE
                            + nbr_node
                        ])
                    elif kind == BC_COPYOUT:
                        f_nbr = f_n
                    elif kind == BC_VLASOV_REFLECT:
                        f_nbr = rebind[Real](q[
                            (phase_e + mirror_elem) * NPE
                            + local_n + mirror_node
                        ])
            out += fmin(c, Real(0.0)) * inv_wP * (f_n - f_nbr)

    rhs[node_base + local_n] = rebind[rhs.ElementType](out)


comptime _REDUCE_BLOCK: Int = 256
"""Threads in the single reduction block. Must be a power of two."""


def _max_abs_fields_kernel(
    fields: RealTensor,
    out_max: RealTensor,
    num_nodes_i64: Int64,
):
    """Single-block reduction: `out_max[c] = max over nodes of
    |fields[node, c]|` for the `VLASOV_AUX_NC` field components. Launch
    with grid_dim=1, block_dim=_REDUCE_BLOCK."""
    comptime NC = VLASOV_AUX_NC
    var num_nodes = Int(num_nodes_i64)
    var partial = stack_allocation[
        WXM_DTYPE, address_space=AddressSpace.SHARED,
    ](row_major[_REDUCE_BLOCK, NC]())

    var tid = Int(thread_idx.x)
    var local = Array[Real, NC](fill=Real(0.0))
    var node = tid
    while node < num_nodes:
        comptime for c in range(NC):
            local[c] = fmax(local[c], fabs(rebind[Real](fields[node * NC + c])))
        node += _REDUCE_BLOCK
    comptime for c in range(NC):
        partial[tid, c] = rebind[partial.ElementType](local[c])
    barrier()

    var active = _REDUCE_BLOCK
    while active > 1:
        active >>= 1
        if tid < active:
            comptime for c in range(NC):
                var lhs = rebind[Real](partial[tid, c])
                var rhs = rebind[Real](partial[tid + active, c])
                partial[tid, c] = rebind[partial.ElementType](fmax(lhs, rhs))
        barrier()

    if tid == 0:
        comptime for c in range(NC):
            out_max[c] = rebind[out_max.ElementType](
                rebind[Real](partial[0, c])
            )


struct DGVlasovSolver[XDIM: Int, VDIM: Int, P_x: Int, P_v: Int]:
    """DG right-hand side and CFL time step for the Vlasov equation on a
    `PhaseMesh[XDIM, VDIM, P_x, P_v]`.

    The fields come from a configuration-space state with the Maxwell
    component order `(E_x, E_y, E_z, B_x, B_y, B_z)`, so a
    `DGState[XDIM, P_x, MAXWELL_NC]` can be passed directly."""

    comptime Phase = PhaseState[
        Self.XDIM, Self.VDIM, Self.P_x, Self.P_v, VLASOV_NC,
    ]
    comptime Fields = DGState[Self.XDIM, Self.P_x, VLASOV_AUX_NC]

    var app: VlasovMaxwell

    def __init__(out self, app: VlasovMaxwell):
        self.app = app

    def compute_rhs(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v],
        mut q: Self.Phase,
        fields: Self.Fields,
        mut rhs: Self.Phase,
    ) raises:
        comptime ND = Self.XDIM + Self.VDIM
        comptime NXN = (Self.P_x + 1) ** Self.XDIM
        comptime NVN = (Self.P_v + 1) ** Self.VDIM
        # The kernel indexes `fields` by configuration element, so a state
        # from another mesh would be read out of bounds.
        if (
            fields.num_elements != mesh.num_x_elements
            or q.num_phase_elements != mesh.num_phase_elements
            or rhs.num_phase_elements != mesh.num_phase_elements
        ):
            raise Error(
                "DGVlasovSolver.compute_rhs: state sizes do not match the "
                + "mesh (fields on " + String(fields.num_elements)
                + " elements, mesh has " + String(mesh.num_x_elements)
                + " configuration elements)"
            )
        # Bring the ghost phase elements of `q` up to date (a mesh shared
        # between ranks only).
        comptime if USE_MPI:
            if mesh.num_ghost > 0:
                if q.num_ghost != mesh.num_ghost:
                    raise Error(
                        "this mesh is shared between ranks: construct phase"
                        + " states and steppers with"
                        + " (ctx, mesh.num_phase_elements, mesh.num_ghost)"
                    )
                exchange_halo[NXN * NVN, Self.P_x + 1, VLASOV_NC, True](
                    ctx, mesh.halo, q.buf, q.alloc_len, q.halo.value(),
                )
        var q_layout = row_major(Int(q.alloc_len))
        var rhs_layout = row_major(Int(rhs.alloc_len))
        var node_v_layout = row_major(Int(mesh.node_v_len))
        var fields_layout = row_major(Int(fields.buf_len))

        # Grid: the last two axes in y and z, the leading ones flattened
        # into x (see the kernel).
        var lead = 1
        for a in range(ND - 2):
            lead *= mesh.n[a]
        var grid_x = lead
        var grid_y = mesh.n[ND - 2]
        var grid_z = mesh.n[ND - 1]
        comptime if ND == 2:
            grid_x = mesh.n[0]
            grid_y = mesh.n[1]
            grid_z = 1

        comptime kernel = _vlasov_rhs_kernel[
            Self.XDIM, Self.VDIM, Self.P_x, Self.P_v,
        ]
        ctx.enqueue_function[kernel](
            TileTensor(q.buf, q_layout),
            TileTensor(rhs.buf, rhs_layout),
            TileTensor(mesh.node_v_buf, node_v_layout),
            TileTensor(fields.buf, fields_layout),
            mesh.kernel_counts(),
            mesh.kernel_bc(),
            mesh.kernel_ghost_first(),
            mesh.kernel_inv_jacobian(),
            self.app.charge_mass_ratio / self.app.skin_depth_norm,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(NXN, NVN, 1),
        )

    def suggest_dt(
        self,
        var ctx: DeviceContext,
        mesh: PhaseMesh[Self.XDIM, Self.VDIM, Self.P_x, Self.P_v],
        fields: Self.Fields,
    ) raises -> Real:
        """Largest stable step at the app's `cfl_max`:

            Δt = cfl_max / ( (2 P_x + 1) Σ_a max|v_a| / Δx_a
                           + (2 P_v + 1) Σ_d max|a_d| / Δv_d ).

        `max|v_a|` is the edge of the velocity mesh; `max|a_d|` is bounded
        from the largest `|E|` and `|B|` components in `fields`, reduced
        on the device so that only six numbers reach the host. The
        read-back still waits for the device queue to drain, so pass a
        `dt_interval` above 1 to `run_until_cfl` when the fields change
        slowly. Does not cover the field solver: take the minimum with
        its own `suggest_dt`."""
        var max_buf = ctx.enqueue_create_buffer[Real.dtype](VLASOV_AUX_NC)
        ctx.enqueue_function[_max_abs_fields_kernel](
            TileTensor(fields.buf, row_major(Int(fields.buf_len))),
            TileTensor(max_buf, row_major(Int(VLASOV_AUX_NC))),
            Int64(fields.total_nodes),
            grid_dim=1, block_dim=_REDUCE_BLOCK,
        )
        var f_max = Array[Real, VLASOV_AUX_NC](fill=Real(0.0))
        with max_buf.map_to_host() as h:
            for c in range(VLASOV_AUX_NC):
                f_max[c] = global_max(h[c])
        var v_max = Array[Real, 3](fill=Real(0.0))
        for d in range(Self.VDIM):
            v_max[d] = mesh.v_abs_max(d)
        var factor = fabs(self.app.charge_mass_ratio / self.app.skin_depth_norm)
        var a_max = Array[Real, 3](fill=Real(0.0))
        a_max[0] = factor * (
            f_max[AUX_EX] + v_max[1] * f_max[AUX_BZ] + v_max[2] * f_max[AUX_BY]
        )
        a_max[1] = factor * (
            f_max[AUX_EY] + v_max[2] * f_max[AUX_BX] + v_max[0] * f_max[AUX_BZ]
        )
        a_max[2] = factor * (
            f_max[AUX_EZ] + v_max[0] * f_max[AUX_BY] + v_max[1] * f_max[AUX_BX]
        )
        var s = Real(0.0)
        for a in range(Self.XDIM):
            s += Real(2 * Self.P_x + 1) * v_max[a] / mesh.dx[a]
        for d in range(Self.VDIM):
            s += Real(2 * Self.P_v + 1) * a_max[d] / mesh.dx[Self.XDIM + d]
        if s < WXM_TINY:
            return WXM_INF
        return self.app.cfl_max / s
