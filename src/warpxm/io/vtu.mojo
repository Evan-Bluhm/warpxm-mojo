# ParaView `.vtu` output for DG fields: one writer for every physics.
#
# `write_vtu(path, mesh, q, names, fields)` takes the list of point-data
# array names and one closure mapping a node state to those values:
#
#     def fields(v: Array[Real, EULER_NC]) {imm} -> Array[Real, 2]:
#         return [v[IDX_RHO], v[IDX_MX] / v[IDX_RHO]]
#     write_vtu(path, mesh, q, ["rho", "u"], fields)
#
# The per-physics presets in `vtu_presets.mojo` are thin wrappers that
# supply the usual primitive variables.
#
# Geometry: each (P+1)^NDIM-node element is split into P^NDIM linear
# `VTK_QUAD` / `VTK_HEXAHEDRON` subcells, so the file shows the DG
# polynomial as a piecewise-linear field at any order. Every element owns
# its own copy of its points, so jumps across element faces are kept.
#
# Encoding: XML header with all arrays in one raw appended-binary section
# (`format="appended"`, UInt64 block headers, little-endian). About 3x
# smaller and far faster to write and load than ASCII, which matters at 3D
# resolutions.
#
# Several ranks: each rank writes its own elements to a piece file
# `<stem>_p<rank>.vtu`, and rank 0 writes `<stem>.pvtu`, the file to open,
# which lists the pieces. On one rank the output is the single `.vtu`.

from std.collections import Array, List
from std.os.path import basename

from warpxm.core.types import Real
from warpxm.mesh.block import BlockMesh
from warpxm.parallel import mpi
from warpxm.variable.state import DGState


comptime VTK_QUAD: Int = 9
comptime VTK_HEXAHEDRON: Int = 12


def _write_block[T: Copyable](mut out: FileHandle, data: List[T], elem_bytes: Int) raises:
    """One appended-data block: UInt64 byte count, then the raw values."""
    var header = List[UInt64]()
    header.append(UInt64(len(data) * elem_bytes))
    out.write_bytes(
        Span[UInt8, _](
            unsafe_ptr=header.unsafe_ptr().unsafe_bitcast[UInt8](), length=8,
        )
    )
    out.write_bytes(
        Span[UInt8, _](
            unsafe_ptr=data.unsafe_ptr().unsafe_bitcast[UInt8](),
            length=len(data) * elem_bytes,
        )
    )


def _array_tag(
    mut out: FileHandle, dtype: String, name: String, components: Int,
    offset: Int,
) raises:
    out.write("        <DataArray type=\"" + dtype + "\"")
    if name != "":
        out.write(" Name=\"" + name + "\"")
    if components > 1:
        out.write(" NumberOfComponents=\"" + String(components) + "\"")
    out.write(" format=\"appended\" offset=\"" + String(offset) + "\"/>\n")


def _write_vtu_file[NDIM: Int, P: Int](
    path: String,
    num_elements: Int,
    points: List[Float64],
    data: List[Float64],
    names: List[String],
) raises:
    """Write the file from gathered host data.

    `points` holds (x, y, z) per node in state order; `data` holds one
    value per (node, field), node-major.
    """
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 ** NDIM
    comptime corners = 2 ** NDIM
    var num_points = num_elements * nodes_per_elem
    var num_cells = num_elements * (P ** NDIM)
    var num_fields = len(names)

    # Connectivity: VTK corner order is counter-clockwise on the bottom
    # face, then the same on the top face.
    var conn = List[Int32](capacity=num_cells * corners)
    for e in range(num_elements):
        var base = e * nodes_per_elem
        comptime if NDIM == 2:
            for j in range(P):
                for i in range(P):
                    conn.append(Int32(base + j * P1 + i))
                    conn.append(Int32(base + j * P1 + i + 1))
                    conn.append(Int32(base + (j + 1) * P1 + i + 1))
                    conn.append(Int32(base + (j + 1) * P1 + i))
        else:
            for k in range(P):
                for j in range(P):
                    for i in range(P):
                        for dk in range(2):
                            var b = base + ((k + dk) * P1 + j) * P1 + i
                            conn.append(Int32(b))
                            conn.append(Int32(b + 1))
                            conn.append(Int32(b + P1 + 1))
                            conn.append(Int32(b + P1))
    var offsets = List[Int32](capacity=num_cells)
    var types = List[UInt8](capacity=num_cells)
    comptime cell_type = VTK_QUAD if NDIM == 2 else VTK_HEXAHEDRON
    for c in range(num_cells):
        offsets.append(Int32(corners * (c + 1)))
        types.append(UInt8(cell_type))

    var out = open(path, "w")
    out.write("<?xml version=\"1.0\"?>\n")
    out.write(
        "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\""
        + " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
    )
    out.write("  <UnstructuredGrid>\n")
    out.write(
        "    <Piece NumberOfPoints=\"" + String(num_points)
        + "\" NumberOfCells=\"" + String(num_cells) + "\">\n"
    )

    # Offsets are measured from the byte after the `_` marker; each block
    # is an 8-byte header plus its payload.
    var offset = 0
    out.write("      <Points>\n")
    _array_tag(out, "Float64", "", 3, offset)
    offset += 8 + num_points * 3 * 8
    out.write("      </Points>\n")
    out.write("      <Cells>\n")
    _array_tag(out, "Int32", "connectivity", 1, offset)
    offset += 8 + len(conn) * 4
    _array_tag(out, "Int32", "offsets", 1, offset)
    offset += 8 + num_cells * 4
    _array_tag(out, "UInt8", "types", 1, offset)
    offset += 8 + num_cells
    out.write("      </Cells>\n")
    if num_fields > 0:
        out.write("      <PointData Scalars=\"" + names[0] + "\">\n")
    else:
        out.write("      <PointData>\n")
    for k in range(num_fields):
        _array_tag(out, "Float64", names[k], 1, offset)
        offset += 8 + num_points * 8
    out.write("      </PointData>\n")
    out.write("    </Piece>\n")
    out.write("  </UnstructuredGrid>\n")
    out.write("  <AppendedData encoding=\"raw\">\n_")

    _write_block(out, points, 8)
    _write_block(out, conn, 4)
    _write_block(out, offsets, 4)
    _write_block(out, types, 1)
    var column = List[Float64](capacity=num_points)
    for k in range(num_fields):
        column.clear()
        for n in range(num_points):
            column.append(data[n * num_fields + k])
        _write_block(out, column, 8)

    out.write("\n  </AppendedData>\n")
    out.write("</VTKFile>\n")
    out.close()


def _stem(path: String) -> String:
    """`path` without its `.vtu` extension."""
    return String(path.removesuffix(".vtu"))


def parallel_vtu_path(path: String) -> String:
    """The file to open for a frame written to `path`: the `.pvtu` index
    on several ranks, `path` itself on one."""
    if mpi.size() == 1:
        return path
    return _stem(path) + ".pvtu"


def _piece_path(path: String, rank: Int) -> String:
    return _stem(path) + "_p" + String(rank) + ".vtu"


def _write_pvtu(path: String, names: List[String]) raises:
    """The index file of a multi-rank frame: array declarations and one
    `Piece` per rank, named relative to the index."""
    var out = open(parallel_vtu_path(path), "w")
    out.write("<?xml version=\"1.0\"?>\n")
    out.write(
        "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\""
        + " byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
    )
    out.write("  <PUnstructuredGrid GhostLevel=\"0\">\n")
    out.write("    <PPoints>\n")
    out.write("      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n")
    out.write("    </PPoints>\n")
    if len(names) > 0:
        out.write("    <PPointData Scalars=\"" + names[0] + "\">\n")
    else:
        out.write("    <PPointData>\n")
    for k in range(len(names)):
        out.write(
            "      <PDataArray type=\"Float64\" Name=\"" + names[k] + "\"/>\n"
        )
    out.write("    </PPointData>\n")
    for r in range(mpi.size()):
        out.write(
            "    <Piece Source=\"" + String(basename(_piece_path(path, r)))
            + "\"/>\n"
        )
    out.write("  </PUnstructuredGrid>\n")
    out.write("</VTKFile>\n")
    out.close()


def write_vtu[
    NDIM: Int, P: Int, NC: Int, NF: Int,
    Fields: def(Array[Real, NC]) -> Array[Real, NF],
](
    path: String,
    mesh: BlockMesh[NDIM, P],
    mut q: DGState[NDIM, P, NC],
    names: List[String],
    fields: Fields,
) raises:
    """Write `fields(q_node)` at every node of a 2D or 3D field as point
    data named by `names` (one name per returned value).

    Call it on every rank with the same `path`; see the header for the
    files a multi-rank run produces."""
    comptime assert NDIM == 2 or NDIM == 3, "VTU output is 2D or 3D"
    debug_assert(len(names) == NF, "one name per field")
    var points = List[Float64](capacity=q.total_nodes * 3)
    var data = List[Float64](capacity=q.total_nodes * NF)
    comptime if NDIM == 2:
        def gather(
            x: Real, y: Real, w: Real, v: Array[Real, NC],
        ) {mut points, mut data, imm}:
            points.append(x)
            points.append(y)
            points.append(0.0)
            var f = fields(v)
            for k in range(NF):
                data.append(f[k])
        mesh.visit_nodes(q, gather)
    else:
        def gather_3d(
            x: Real, y: Real, z: Real, w: Real, v: Array[Real, NC],
        ) {mut points, mut data, imm}:
            points.append(x)
            points.append(y)
            points.append(z)
            var f = fields(v)
            for k in range(NF):
                data.append(f[k])
        mesh.visit_nodes(q, gather_3d)
    if mpi.size() == 1:
        _write_vtu_file[NDIM, P](path, mesh.num_elements, points, data, names)
    else:
        _write_vtu_file[NDIM, P](
            _piece_path(path, mpi.rank()), mesh.num_elements, points, data,
            names,
        )
        if mpi.is_root():
            _write_pvtu(path, names)
