# 3D Maxwell VTU frame writer for `DGState3D` with `MAXWELL_NC` components.
#
# Tensor-product extension of `vtu_maxwell_2d.mojo`. Each (P+1)³-node hex
# element is subdivided into P³ linear `VTK_HEXAHEDRON` (type 12) subcells so
# the writer is order-agnostic. Dumps:
#     E_x, E_y, E_z, B_x, B_y, B_z   (raw field components)
#     E_energy = ½|E|²,  B_energy = ½|B|²  (per-node energy densities)

from max.gpu.host import DeviceContext

from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_3d import DGState3D


comptime VTK_HEXAHEDRON: Int = 12


def dump_maxwell_3d_frame[P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh3D[P],
    q: DGState3D[P, MAXWELL_NC],
) raises:
    """Write a 3D Maxwell field at `path` as a ParaView-readable `.vtu`."""
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1 * P1
    comptime subhexes_per_elem = P * P * P

    var total_points = mesh.num_elements * nodes_per_elem
    var total_cells = mesh.num_elements * subhexes_per_elem

    var nodes_host = ctx.enqueue_create_host_buffer[Real.dtype](
        mesh.num_elements * nodes_per_elem * 3
    )
    ctx.enqueue_copy(dst_buf=nodes_host, src_buf=mesh.node_xyz_buf)
    var q_host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    ctx.enqueue_copy(dst_buf=q_host, src_buf=q.buf)
    ctx.synchronize()

    var out = open(path, "w")

    out.write("<?xml version=\"1.0\"?>\n")
    out.write("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n")
    out.write("  <UnstructuredGrid>\n")
    out.write("    <Piece NumberOfPoints=\"")
    out.write(String(total_points))
    out.write("\" NumberOfCells=\"")
    out.write(String(total_cells))
    out.write("\">\n")

    # --- Points -------------------------------------------------------------
    out.write("      <Points>\n")
    out.write("        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var idx = (e * nodes_per_elem + n) * 3
            out.write("          ")
            out.write(String(nodes_host[idx + 0]))
            out.write(" ")
            out.write(String(nodes_host[idx + 1]))
            out.write(" ")
            out.write(String(nodes_host[idx + 2]))
            out.write("\n")
    out.write("        </DataArray>\n")
    out.write("      </Points>\n")

    # --- Cell connectivity --------------------------------------------------
    @always_inline
    def _node_offset(i: Int, j: Int, k: Int) -> Int:
        return k * P1 * P1 + j * P1 + i

    out.write("      <Cells>\n")
    out.write("        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        var base = e * nodes_per_elem
        for k_sub in range(P):
            for j_sub in range(P):
                for i_sub in range(P):
                    var p0 = base + _node_offset(i_sub, j_sub, k_sub)
                    var p1 = base + _node_offset(i_sub + 1, j_sub, k_sub)
                    var p2 = base + _node_offset(i_sub + 1, j_sub + 1, k_sub)
                    var p3 = base + _node_offset(i_sub, j_sub + 1, k_sub)
                    var p4 = base + _node_offset(i_sub, j_sub, k_sub + 1)
                    var p5 = base + _node_offset(i_sub + 1, j_sub, k_sub + 1)
                    var p6 = base + _node_offset(i_sub + 1, j_sub + 1, k_sub + 1)
                    var p7 = base + _node_offset(i_sub, j_sub + 1, k_sub + 1)
                    out.write("          ")
                    out.write(String(p0)); out.write(" ")
                    out.write(String(p1)); out.write(" ")
                    out.write(String(p2)); out.write(" ")
                    out.write(String(p3)); out.write(" ")
                    out.write(String(p4)); out.write(" ")
                    out.write(String(p5)); out.write(" ")
                    out.write(String(p6)); out.write(" ")
                    out.write(String(p7)); out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n")
    for c in range(total_cells):
        out.write("          ")
        out.write(String(8 * (c + 1)))
        out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n")
    for _ in range(total_cells):
        out.write("          ")
        out.write(String(VTK_HEXAHEDRON))
        out.write("\n")
    out.write("        </DataArray>\n")
    out.write("      </Cells>\n")

    # --- Point data ---------------------------------------------------------
    out.write("      <PointData Scalars=\"E_x\">\n")

    out.write("        <DataArray type=\"Float64\" Name=\"E_x\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            out.write("          "); out.write(String(q_host[b + IDX_EX])); out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"E_y\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            out.write("          "); out.write(String(q_host[b + IDX_EY])); out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"E_z\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            out.write("          "); out.write(String(q_host[b + IDX_EZ])); out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"B_x\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            out.write("          "); out.write(String(q_host[b + IDX_BX])); out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"B_y\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            out.write("          "); out.write(String(q_host[b + IDX_BY])); out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"B_z\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MAXWELL_NC
            out.write("          "); out.write(String(q_host[b + IDX_BZ])); out.write("\n")
    out.write("        </DataArray>\n")

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
