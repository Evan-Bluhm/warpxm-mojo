# 2D VTU frame writer for ideal MHD / MixedGLM-MHD fields.
#
# Same geometric decomposition as `vtu_2d.mojo` (each (P+1)² element →
# P² linear VTK_QUAD subcells). Point data: ρ, u, v, P_gas, B_x, B_y,
# B_mag² = |B|² / 2 (magnetic pressure), and ψ if present (GLM).
#
# Works for both `MHD_NC = 8` (plain ideal MHD) and `MHD_NC = 9`
# (MixedGLM). The 9-component path writes an extra ψ field; otherwise
# the per-frame data are identical.

from std.gpu.host import DeviceContext

from warpxm.apps.imhd_2d import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.variable.state_2d import DGState2D


comptime VTK_QUAD: Int = 9


def _write_imhd_2d_xml[NC: Int, P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh2D[P],
    q: DGState2D[P, NC],
    gamma: Real,
    has_psi: Bool,
) raises:
    """Underlying writer — `NC` is 8 (plain IMHD) or 9 (GLM)."""
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
    out.write("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n")
    out.write("  <UnstructuredGrid>\n")
    out.write("    <Piece NumberOfPoints=\"")
    out.write(String(total_points))
    out.write("\" NumberOfCells=\"")
    out.write(String(total_cells))
    out.write("\">\n")

    out.write("      <Points>\n")
    out.write("        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n")
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

    out.write("      <PointData Scalars=\"rho\">\n")

    # Helper to write a single scalar field from a callable on (rho, mx, my, mz, e, Bx, By, Bz).
    # In Mojo we just inline the loops per field for clarity.

    # rho
    out.write("        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * NC
            out.write("          ")
            out.write(String(q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    # u, v
    for comp_pair in [(IDX_MX, "u"), (IDX_MY, "v")]:
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
            out.write("          ")
            out.write(String(pp))
            out.write("\n")
    out.write("        </DataArray>\n")

    # B_x, B_y
    for comp_pair in [(IDX_BX, "Bx"), (IDX_BY, "By")]:
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

    # |B|² / 2 (magnetic pressure)
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

    # ψ (GLM) — only present at NC = 9.
    if has_psi:
        out.write("        <DataArray type=\"Float64\" Name=\"psi\" format=\"ascii\">\n")
        for e in range(mesh.num_elements):
            for n in range(nodes_per_elem):
                var b = (e * nodes_per_elem + n) * NC
                # IDX_PSI == 8 by definition in the GLM app; we hardcode here
                # to avoid pulling in the GLM constants when this path is
                # used with NC=8 IMHD.
                out.write("          ")
                out.write(String(q_host[b + 8]))
                out.write("\n")
        out.write("        </DataArray>\n")

    out.write("      </PointData>\n")
    out.write("    </Piece>\n")
    out.write("  </UnstructuredGrid>\n")
    out.write("</VTKFile>\n")


def dump_imhd_2d_frame[P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh2D[P],
    q: DGState2D[P, MHD_NC],
    gamma: Real,
) raises:
    """Dump an 8-component ideal-MHD field to `path`."""
    _write_imhd_2d_xml[MHD_NC, P](path, ctx, mesh, q, gamma, has_psi=False)


def dump_imhd_glm_2d_frame[P: Int, NC: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh2D[P],
    q: DGState2D[P, NC],
    gamma: Real,
) raises:
    """Dump a 9-component GLM-MHD field (ψ included) to `path`."""
    _write_imhd_2d_xml[NC, P](path, ctx, mesh, q, gamma, has_psi=True)
