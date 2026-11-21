# 2D VTU frame writer for Maxwell fields.
#
# Same per-element decomposition as the Euler / MHD writers: each
# `(P+1)²`-node quad element is subdivided into P² linear VTK_QUAD
# subcells. Point data: E_x, E_y, E_z, B_x, B_y, B_z, plus the energy
# densities `½|E|²` (electric) and `½|B|²` (magnetic) for easy display.

from max.gpu.host import DeviceContext

from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.variable.state_2d import DGState2D


comptime VTK_QUAD: Int = 9


def dump_maxwell_2d_frame[P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh2D[P],
    q: DGState2D[P, MAXWELL_NC],
) raises:
    """Write a Maxwell 2D field at `path` as a ParaView-readable `.vtu`."""
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

    # --- Cells --------------------------------------------------------------
    out.write("      <Cells>\n")
    out.write("        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        var base = e * nodes_per_elem
        for j_sub in range(P):
            for i_sub in range(P):
                var p0 = base + j_sub * P1 + i_sub
                var p1 = base + j_sub * P1 + (i_sub + 1)
                var p2 = base + (j_sub + 1) * P1 + (i_sub + 1)
                var p3 = base + (j_sub + 1) * P1 + i_sub
                out.write("          ")
                out.write(String(p0)); out.write(" ")
                out.write(String(p1)); out.write(" ")
                out.write(String(p2)); out.write(" ")
                out.write(String(p3)); out.write("\n")
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

    # --- Point data --------------------------------------------------------
    out.write("      <PointData Scalars=\"E_z\">\n")

    # E_x
    out.write("        <DataArray type=\"Float64\" Name=\"E_x\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            out.write("          ")
            out.write(String(q_host[(e * nodes_per_elem + n) * MAXWELL_NC + IDX_EX]))
            out.write("\n")
    out.write("        </DataArray>\n")
    # E_y
    out.write("        <DataArray type=\"Float64\" Name=\"E_y\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            out.write("          ")
            out.write(String(q_host[(e * nodes_per_elem + n) * MAXWELL_NC + IDX_EY]))
            out.write("\n")
    out.write("        </DataArray>\n")
    # E_z
    out.write("        <DataArray type=\"Float64\" Name=\"E_z\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            out.write("          ")
            out.write(String(q_host[(e * nodes_per_elem + n) * MAXWELL_NC + IDX_EZ]))
            out.write("\n")
    out.write("        </DataArray>\n")
    # B_x
    out.write("        <DataArray type=\"Float64\" Name=\"B_x\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            out.write("          ")
            out.write(String(q_host[(e * nodes_per_elem + n) * MAXWELL_NC + IDX_BX]))
            out.write("\n")
    out.write("        </DataArray>\n")
    # B_y
    out.write("        <DataArray type=\"Float64\" Name=\"B_y\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            out.write("          ")
            out.write(String(q_host[(e * nodes_per_elem + n) * MAXWELL_NC + IDX_BY]))
            out.write("\n")
    out.write("        </DataArray>\n")
    # B_z
    out.write("        <DataArray type=\"Float64\" Name=\"B_z\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            out.write("          ")
            out.write(String(q_host[(e * nodes_per_elem + n) * MAXWELL_NC + IDX_BZ]))
            out.write("\n")
    out.write("        </DataArray>\n")

    # ½|E|² (electric energy density)
    out.write("        <DataArray type=\"Float64\" Name=\"E_energy\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            var Ex = q_host[b + IDX_EX]
            var Ey = q_host[b + IDX_EY]
            var Ez = q_host[b + IDX_EZ]
            out.write("          ")
            out.write(String(Real(0.5) * (Ex * Ex + Ey * Ey + Ez * Ez)))
            out.write("\n")
    out.write("        </DataArray>\n")

    # ½|B|² (magnetic energy density)
    out.write("        <DataArray type=\"Float64\" Name=\"B_energy\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            var Bx = q_host[b + IDX_BX]
            var By = q_host[b + IDX_BY]
            var Bz = q_host[b + IDX_BZ]
            out.write("          ")
            out.write(String(Real(0.5) * (Bx * Bx + By * By + Bz * Bz)))
            out.write("\n")
    out.write("        </DataArray>\n")

    out.write("      </PointData>\n")
    out.write("    </Piece>\n")
    out.write("  </UnstructuredGrid>\n")
    out.write("</VTKFile>\n")
