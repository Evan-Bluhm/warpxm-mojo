# 3D VTU frame writer for ideal MHD / MixedGLM-MHD fields.
#
# Tensor-product extension of `vtu_imhd_2d.mojo`. Each (P+1)³-node element
# becomes P³ linear `VTK_HEXAHEDRON` (cell type 12) subcells. Point data:
# ρ, u, v, w, P_gas, B_x, B_y, B_z, |B|² / 2, and ψ if present (GLM).
#
# Works for both `MHD_NC = 8` (plain ideal MHD) and `MHD_NC = 9` (GLM).

from std.gpu.host import DeviceContext

from warpxm.apps.imhd import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_3d import DGState3D


comptime VTK_HEXAHEDRON: Int = 12


def _write_imhd_3d_xml[NC: Int, P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh3D[P],
    q: DGState3D[P, NC],
    gamma: Real,
    has_psi: Bool,
) raises:
    """Underlying writer — `NC` is 8 (plain IMHD) or 9 (GLM)."""
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

    # --- Cells (P³ linear hexes per element) --------------------------------
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
    out.write("      <PointData Scalars=\"rho\">\n")

    # ρ
    out.write("        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * NC
            out.write("          "); out.write(String(q_host[b + IDX_RHO])); out.write("\n")
    out.write("        </DataArray>\n")

    # u, v, w
    for comp_pair in [(IDX_MX, "u"), (IDX_MY, "v"), (IDX_MZ, "w")]:
        out.write("        <DataArray type=\"Float64\" Name=\"")
        out.write(comp_pair[1])
        out.write("\" format=\"ascii\">\n")
        for e in range(mesh.num_elements):
            for n in range(nodes_per_elem):
                var b = (e * nodes_per_elem + n) * NC
                out.write("          ")
                out.write(String(q_host[b + comp_pair[0]] / q_host[b + IDX_RHO]))
                out.write("\n")
        out.write("        </DataArray>\n")

    # P_gas
    out.write("        <DataArray type=\"Float64\" Name=\"P\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * NC
            var rho = q_host[b + IDX_RHO]
            var ux = q_host[b + IDX_MX] / rho
            var uy = q_host[b + IDX_MY] / rho
            var uz = q_host[b + IDX_MZ] / rho
            var Bx = q_host[b + IDX_BX]
            var By = q_host[b + IDX_BY]
            var Bz = q_host[b + IDX_BZ]
            var ke = Real(0.5) * rho * (ux * ux + uy * uy + uz * uz)
            var mag = Real(0.5) * (Bx * Bx + By * By + Bz * Bz)
            var pp = (gamma - Real(1.0)) * (q_host[b + IDX_E] - ke - mag)
            out.write("          "); out.write(String(pp)); out.write("\n")
    out.write("        </DataArray>\n")

    # B_x, B_y, B_z
    for comp_pair in [(IDX_BX, "Bx"), (IDX_BY, "By"), (IDX_BZ, "Bz")]:
        out.write("        <DataArray type=\"Float64\" Name=\"")
        out.write(comp_pair[1])
        out.write("\" format=\"ascii\">\n")
        for e in range(mesh.num_elements):
            for n in range(nodes_per_elem):
                var b = (e * nodes_per_elem + n) * NC
                out.write("          ")
                out.write(String(q_host[b + comp_pair[0]]))
                out.write("\n")
        out.write("        </DataArray>\n")

    # |B|² / 2
    out.write("        <DataArray type=\"Float64\" Name=\"B2_over_2\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * NC
            var Bx = q_host[b + IDX_BX]
            var By = q_host[b + IDX_BY]
            var Bz = q_host[b + IDX_BZ]
            out.write("          ")
            out.write(String(Real(0.5) * (Bx * Bx + By * By + Bz * Bz)))
            out.write("\n")
    out.write("        </DataArray>\n")

    # ψ (GLM)
    if has_psi:
        out.write("        <DataArray type=\"Float64\" Name=\"psi\" format=\"ascii\">\n")
        for e in range(mesh.num_elements):
            for n in range(nodes_per_elem):
                var b = (e * nodes_per_elem + n) * NC
                # IDX_PSI == 8 in the GLM app.
                out.write("          ")
                out.write(String(q_host[b + 8]))
                out.write("\n")
        out.write("        </DataArray>\n")

    out.write("      </PointData>\n")
    out.write("    </Piece>\n")
    out.write("  </UnstructuredGrid>\n")
    out.write("</VTKFile>\n")


def dump_imhd_3d_frame[P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh3D[P],
    q: DGState3D[P, MHD_NC],
    gamma: Real,
) raises:
    """Dump an 8-component 3D ideal-MHD field to `path`."""
    _write_imhd_3d_xml[MHD_NC, P](path, ctx, mesh, q, gamma, has_psi=False)


def dump_imhd_glm_3d_frame[P: Int, NC: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh3D[P],
    q: DGState3D[P, NC],
    gamma: Real,
) raises:
    """Dump a 9-component 3D GLM-MHD field (ψ included) to `path`."""
    _write_imhd_3d_xml[NC, P](path, ctx, mesh, q, gamma, has_psi=True)
