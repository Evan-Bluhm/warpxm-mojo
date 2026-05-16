# 2D VTU frame writer for `DGState2D` fields.
#
# Dumps an XML VTK UnstructuredGrid (`.vtu`) file viewable in ParaView. ASCII
# format keeps the writer trivial — switching to appended-raw binary later is
# straightforward.
#
# Per-element decomposition: each `(P+1)²`-node quad element gets subdivided
# into `P²` linear `VTK_QUAD` (cell type 9) subcells, so the writer is
# order-agnostic and renders the DG variation as a piecewise-linear field on
# a fine sub-mesh. Each element owns its own copy of the (P+1)² geometry
# points — the file is DG-discontinuous by construction, so a contact /
# shock cell shows correctly without averaging across element boundaries.

from std.gpu.host import DeviceContext

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_E
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.variable.state_2d import DGState2D


comptime VTK_QUAD: Int = 9


def dump_euler_2d_frame[P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh2D[P],
    q: DGState2D[P, EULER_NC],
    gamma: Real,
) raises:
    """Write an Euler 2D field at `path` as a ParaView-readable `.vtu`.

    Dumps the (ρ, u, v, P) primitive scalars at every DG node, with each
    element subdivided into P² linear quads for connectivity. The field
    set is hardcoded to the 5-moment Euler primitive layout; sibling
    writers (`vtu_imhd_2d.mojo`, `vtu_maxwell_2d.mojo`, and the 3D
    family) follow the same per-physics pattern with their own field
    lists.
    """
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1
    comptime subquads_per_elem = P * P

    var total_points = mesh.num_elements * nodes_per_elem
    var total_cells = mesh.num_elements * subquads_per_elem

    var nodes_host = ctx.enqueue_create_host_buffer[Real.dtype](
        mesh.num_elements * nodes_per_elem * 2
    )
    ctx.enqueue_copy(dst_buf=nodes_host, src_buf=mesh.node_xy_buf)
    var q_host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    ctx.enqueue_copy(dst_buf=q_host, src_buf=q.buf)
    ctx.synchronize()

    var out = open(path, "w")

    out.write("<?xml version=\"1.0\"?>\n")
    out.write(
        "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
    )
    out.write("  <UnstructuredGrid>\n")
    out.write("    <Piece NumberOfPoints=\"")
    out.write(String(total_points))
    out.write("\" NumberOfCells=\"")
    out.write(String(total_cells))
    out.write("\">\n")

    # --- Points -------------------------------------------------------------
    out.write("      <Points>\n")
    out.write(
        "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n"
    )
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var idx = (e * nodes_per_elem + n) * 2
            out.write("          ")
            out.write(String(nodes_host[idx + 0]))
            out.write(" ")
            out.write(String(nodes_host[idx + 1]))
            out.write(" 0.0\n")
    out.write("        </DataArray>\n")
    out.write("      </Points>\n")

    # --- Cell connectivity --------------------------------------------------
    out.write("      <Cells>\n")
    out.write("        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        var base = e * nodes_per_elem
        for j_sub in range(P):
            for i_sub in range(P):
                # CCW quad: BL → BR → TR → TL.
                var p0 = base + j_sub * P1 + i_sub
                var p1 = base + j_sub * P1 + (i_sub + 1)
                var p2 = base + (j_sub + 1) * P1 + (i_sub + 1)
                var p3 = base + (j_sub + 1) * P1 + i_sub
                out.write("          ")
                out.write(String(p0))
                out.write(" ")
                out.write(String(p1))
                out.write(" ")
                out.write(String(p2))
                out.write(" ")
                out.write(String(p3))
                out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n")
    for c in range(total_cells):
        out.write("          ")
        out.write(String(4 * (c + 1)))
        out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n")
    for _ in range(total_cells):
        out.write("          ")
        out.write(String(VTK_QUAD))
        out.write("\n")
    out.write("        </DataArray>\n")
    out.write("      </Cells>\n")

    # --- Point data (Euler scalars: ρ, u, v, P) -----------------------------
    out.write("      <PointData Scalars=\"rho\">\n")

    # rho
    out.write("        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            out.write("          ")
            out.write(String(q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    # u
    out.write("        <DataArray type=\"Float64\" Name=\"u\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            out.write("          ")
            out.write(String(q_host[b + IDX_MX] / q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    # v
    out.write("        <DataArray type=\"Float64\" Name=\"v\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            out.write("          ")
            out.write(String(q_host[b + IDX_MY] / q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    # P
    out.write("        <DataArray type=\"Float64\" Name=\"P\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            var rho = q_host[b + IDX_RHO]
            var u = q_host[b + IDX_MX] / rho
            var v = q_host[b + IDX_MY] / rho
            var et = q_host[b + IDX_E]
            var ke = Real(0.5) * rho * (u * u + v * v)
            var pp = (gamma - Real(1.0)) * (et - ke)
            out.write("          ")
            out.write(String(pp))
            out.write("\n")
    out.write("        </DataArray>\n")

    out.write("      </PointData>\n")
    out.write("    </Piece>\n")
    out.write("  </UnstructuredGrid>\n")
    out.write("</VTKFile>\n")
