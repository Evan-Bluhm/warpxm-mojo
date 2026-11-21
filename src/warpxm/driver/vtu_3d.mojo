# 3D VTU frame writer for `DGState3D` Euler fields.
#
# Tensor-product extension of `vtu_2d.mojo`. Each (P+1)³-node hex element
# gets subdivided into P³ linear `VTK_HEXAHEDRON` (cell type 12) subcells,
# so the writer is order-agnostic and renders the DG variation as a
# piecewise-trilinear field on a fine sub-mesh.
#
# Each element owns its own copy of the (P+1)³ geometry points — the file
# is DG-discontinuous by construction, so element-boundary jumps show
# correctly without averaging.
#
# Point data dumped from an Euler state: ρ, u, v, w, P.

from max.gpu.host import DeviceContext

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_3d import DGState3D


comptime VTK_HEXAHEDRON: Int = 12


def dump_euler_3d_frame[P: Int](
    path: String,
    var ctx: DeviceContext,
    mesh: BlockMesh3D[P],
    q: DGState3D[P, EULER_NC],
    gamma: Real,
) raises:
    """Write a 3D Euler field at `path` as a ParaView-readable `.vtu`."""
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
    # VTK_HEXAHEDRON corner ordering (cell type 12):
    #   0 = bottom-front-left   (i,   j,   k  )
    #   1 = bottom-front-right  (i+1, j,   k  )
    #   2 = bottom-back-right   (i+1, j+1, k  )
    #   3 = bottom-back-left    (i,   j+1, k  )
    #   4 = top-front-left      (i,   j,   k+1)
    #   5 = top-front-right     (i+1, j,   k+1)
    #   6 = top-back-right      (i+1, j+1, k+1)
    #   7 = top-back-left       (i,   j+1, k+1)

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

    # --- Point data (ρ, u, v, w, P) -----------------------------------------
    out.write("      <PointData Scalars=\"rho\">\n")

    out.write("        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            out.write("          ")
            out.write(String(q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"u\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            out.write("          ")
            out.write(String(q_host[b + IDX_MX] / q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"v\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            out.write("          ")
            out.write(String(q_host[b + IDX_MY] / q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"w\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            out.write("          ")
            out.write(String(q_host[b + IDX_MZ] / q_host[b + IDX_RHO]))
            out.write("\n")
    out.write("        </DataArray>\n")

    out.write("        <DataArray type=\"Float64\" Name=\"P\" format=\"ascii\">\n")
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            var rho = q_host[b + IDX_RHO]
            var u = q_host[b + IDX_MX] / rho
            var v = q_host[b + IDX_MY] / rho
            var w = q_host[b + IDX_MZ] / rho
            var et = q_host[b + IDX_E]
            var ke = Real(0.5) * rho * (u * u + v * v + w * w)
            var pp = (gamma - Real(1.0)) * (et - ke)
            out.write("          ")
            out.write(String(pp))
            out.write("\n")
    out.write("        </DataArray>\n")

    out.write("      </PointData>\n")
    out.write("    </Piece>\n")
    out.write("  </UnstructuredGrid>\n")
    out.write("</VTKFile>\n")
