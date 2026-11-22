# Round-trip test for every VTU frame writer and the PVD frame collection.
#
# Covers the generic writer `io/vtu` through every preset in
# `io/vtu_presets` (Euler, ideal MHD, GLM-MHD, Maxwell; 2D and 3D) and
# `io/frame_writer`.
#
# Each writer is handed a state built from a known analytic primitive
# field on a mesh with unequal element counts and extents (so a swapped
# axis or stride shows up). The file is then read back and parsed:
#
#   * exactly one well-formed VTKFile / Piece, self-closing DataArray tags,
#     and exactly the expected set of point-data arrays;
#   * declared NumberOfPoints / NumberOfCells equal the mesh counts AND
#     the number of values actually present in every array;
#   * point coordinates equal the DG node coordinates, in node order;
#   * connectivity indices are in range, every sub-cell has positive
#     orientation and the right corner layout, and the sub-cell measures
#     sum to the domain area / volume; offsets and cell types are right;
#   * every point-data array equals the analytic value at its point
#     (derived quantities such as u = m/ρ and P included) to 1e-14 — the
#     arrays are raw little-endian Float64, so only arithmetic round-off
#     is left;
#   * the appended-binary section is exactly the declared blocks: every
#     array's type, offset and byte count is checked against the mesh.
#
# Output goes to a directory owned by this test; any file left by an
# earlier run is deleted before it is rewritten, so a writer that
# silently writes nothing cannot pass on a stale file.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vtu_writers.mojo

from std.collections import List
from std.collections.string import atol
from std.memory import bitcast
from std.os import makedirs, remove, rmdir
from std.os.path import exists
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from warpxm.apps.euler import EULER_NC
from warpxm.apps.imhd import MHD_NC
from warpxm.apps.imhd_glm import MHD_GLM_NC
from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.io.frame_writer import FrameWriter
from warpxm.io.vtu_presets import dump_euler_frame
from warpxm.io.vtu_presets import dump_euler_frame
from warpxm.io.vtu_presets import dump_imhd_frame, dump_imhd_glm_frame
from warpxm.io.vtu_presets import dump_imhd_frame, dump_imhd_glm_frame
from warpxm.io.vtu_presets import dump_maxwell_frame
from warpxm.io.vtu_presets import dump_maxwell_frame
from warpxm.mesh.block import BlockMesh
from warpxm.variable.state import DGState


comptime OUTPUT_DIR: String = "/tmp/warpxm_test_vtu_writers"
comptime GAMMA: Real = 1.4

# Unequal counts and extents on purpose.
comptime NX: Int = 3
comptime NY: Int = 2
comptime NZ: Int = 4
comptime X_MIN: Real = -0.5
comptime X_MAX: Real = 1.0
comptime Y_MIN: Real = 0.25
comptime Y_MAX: Real = 1.25
comptime Z_MIN: Real = 0.0
comptime Z_MAX: Real = 0.5

comptime KIND_FLUID: Int = 0     # (ρ, m, E, B, ψ) truncated to NC
comptime KIND_MAXWELL: Int = 1   # (E, B)

comptime VALUE_TOL: Real = 1.0e-14
comptime COORD_TOL: Real = 1.0e-14


# --- analytic field -------------------------------------------------------

def _field(name: String, x: Real, y: Real, z: Real, shift: Real) raises -> Real:
    """Analytic value of the named output field at a point. Every field is
    a different non-separable polynomial; `shift` distinguishes frames."""
    var rho = Real(1.0) + Real(0.2) * x + Real(0.3) * y * y + Real(0.1) * z + shift
    var ux = Real(0.5) + x * y
    var uy = Real(-0.3) + Real(0.7) * y + z
    var uz = Real(0.2) + Real(0.4) * x - Real(0.1) * z
    var pp = Real(2.0) + x - Real(0.5) * y + Real(0.25) * z * x
    var bx = Real(0.3) + y
    var by = Real(-0.2) + x * x
    var bz = Real(0.6) - z + Real(0.5) * y
    var psi = Real(0.05) + Real(0.1) * x * y
    if name == "rho": return rho
    if name == "u": return ux
    if name == "v": return uy
    if name == "w": return uz
    if name == "P": return pp
    if name == "Bx" or name == "B_x": return bx
    if name == "By" or name == "B_y": return by
    if name == "Bz" or name == "B_z": return bz
    if name == "psi": return psi
    if name == "B2_over_2" or name == "B_energy":
        return Real(0.5) * (bx * bx + by * by + bz * bz)
    # Maxwell reuses the velocity polynomials for E.
    if name == "E_x": return ux
    if name == "E_y": return uy
    if name == "E_z": return uz
    if name == "E_energy":
        return Real(0.5) * (ux * ux + uy * uy + uz * uz)
    raise Error("unknown field " + name)


def _conserved(
    kind: Int, has_b: Bool, x: Real, y: Real, z: Real, shift: Real,
) raises -> List[Real]:
    """Solver-layout state vector at a point (longest layout; the caller
    stores the first NC entries)."""
    var out = List[Real]()
    if kind == KIND_MAXWELL:
        for name in ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z"]:
            out.append(_field(name, x, y, z, shift))
        return out^
    var rho = _field("rho", x, y, z, shift)
    var ux = _field("u", x, y, z, shift)
    var uy = _field("v", x, y, z, shift)
    var uz = _field("w", x, y, z, shift)
    var energy = (
        _field("P", x, y, z, shift) / (GAMMA - Real(1.0))
        + Real(0.5) * rho * (ux * ux + uy * uy + uz * uz)
    )
    # The Euler state carries no field energy.
    if has_b:
        energy += _field("B2_over_2", x, y, z, shift)
    out.append(rho)
    out.append(rho * ux)
    out.append(rho * uy)
    out.append(rho * uz)
    out.append(energy)
    out.append(_field("Bx", x, y, z, shift))
    out.append(_field("By", x, y, z, shift))
    out.append(_field("Bz", x, y, z, shift))
    out.append(_field("psi", x, y, z, shift))
    return out^


def _fill_2d[P: Int, NC: Int](
    var ctx: DeviceContext, mut q: DGState[2, P, NC], kind: Int, shift: Real,
) raises -> List[Real]:
    """Fill `q` from the analytic field; return node coordinates (x, y, z
    per node, in solver node order) computed independently of the mesh's
    own coordinate buffer."""
    var P1 = P + 1
    var ref_nodes = NodalBasis1D[P].nodes()
    var dx = (X_MAX - X_MIN) / Real(NX)
    var dy = (Y_MAX - Y_MIN) / Real(NY)
    var coords = List[Real]()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    var k = 0
    for j in range(NY):
        for i in range(NX):
            for jn in range(P1):
                var y = Y_MIN + Real(j) * dy + (Real(1.0) + ref_nodes[jn]) * Real(0.5) * dy
                for in_ in range(P1):
                    var x = X_MIN + Real(i) * dx + (Real(1.0) + ref_nodes[in_]) * Real(0.5) * dx
                    coords.append(x)
                    coords.append(y)
                    coords.append(Real(0.0))
                    var u = _conserved(kind, NC > EULER_NC, x, y, Real(0.0), shift)
                    for c in range(NC):
                        host[k * NC + c] = u[c]
                    k += 1
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()
    return coords^


def _fill_3d[P: Int, NC: Int](
    var ctx: DeviceContext, mut q: DGState[3, P, NC], kind: Int,
) raises -> List[Real]:
    var P1 = P + 1
    var ref_nodes = NodalBasis1D[P].nodes()
    var dx = (X_MAX - X_MIN) / Real(NX)
    var dy = (Y_MAX - Y_MIN) / Real(NY)
    var dz = (Z_MAX - Z_MIN) / Real(NZ)
    var coords = List[Real]()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    var k = 0
    for kz in range(NZ):
        for j in range(NY):
            for i in range(NX):
                for kn in range(P1):
                    var z = Z_MIN + Real(kz) * dz + (Real(1.0) + ref_nodes[kn]) * Real(0.5) * dz
                    for jn in range(P1):
                        var y = Y_MIN + Real(j) * dy + (Real(1.0) + ref_nodes[jn]) * Real(0.5) * dy
                        for in_ in range(P1):
                            var x = X_MIN + Real(i) * dx + (Real(1.0) + ref_nodes[in_]) * Real(0.5) * dx
                            coords.append(x)
                            coords.append(y)
                            coords.append(z)
                            var u = _conserved(kind, NC > EULER_NC, x, y, z, Real(0.0))
                            for c in range(NC):
                                host[k * NC + c] = u[c]
                            k += 1
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()
    return coords^


# --- file helpers ---------------------------------------------------------

def _fresh(path: String) raises -> String:
    """Delete any stale copy of `path`; return it."""
    if exists(path):
        remove(path)
    assert_true(not exists(path), "could not remove stale " + path)
    return path


struct VtuFile(Movable):
    """A parsed `.vtu`: the XML text (header and trailer, with the binary
    section cut out) and the raw appended-data bytes."""
    var text: String
    var blob: List[UInt8]

    def __init__(out self, var text: String, var blob: List[UInt8]):
        self.text = text^
        self.blob = blob^


def _read_text(path: String) raises -> String:
    assert_true(exists(path), "writer produced no file at " + path)
    return String(open(path, "r").read())


def _find(bytes: List[UInt8], pattern: String, start: Int) -> Int:
    """First index >= start where `pattern` occurs in `bytes`, or -1."""
    var pat = pattern.as_bytes()
    var n = len(pat)
    var i = start
    while i + n <= len(bytes):
        var ok = True
        for k in range(n):
            if bytes[i + k] != pat[k]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _read(path: String) raises -> VtuFile:
    """Split a `.vtu` at its appended-data section."""
    assert_true(exists(path), "writer produced no file at " + path)
    var bytes = open(path, "r").read_bytes()
    var open_tag = "<AppendedData encoding=\"raw\">\n_"
    var at = _find(bytes, open_tag, 0)
    assert_true(at > 0, "no raw AppendedData section in " + path)
    var begin = at + open_tag.byte_length()
    # The binary payload can contain anything, so find the trailer from
    # the end of the file.
    var trailer = "\n  </AppendedData>\n</VTKFile>\n"
    var end = len(bytes) - trailer.byte_length()
    assert_true(end >= begin and _find(bytes, trailer, end) == end,
                "bad trailer in " + path)
    var view = Span[UInt8, _](bytes)
    var text = (
        String(from_utf8_lossy=view[0:begin])
        + String(from_utf8_lossy=view[end:len(bytes)])
    )
    var blob = List[UInt8](capacity=end - begin)
    for i in range(begin, end):
        blob.append(bytes[i])
    return VtuFile(text^, blob^)


def _abs(v: Real) -> Real:
    return v if v >= Real(0.0) else -v


def _attr(tag: String, key: String) raises -> String:
    var marker = key + "=\""
    assert_true(tag.count(marker) == 1, "expected exactly one " + key + " in " + tag)
    var begin = tag.find(marker) + marker.byte_length()
    var end = tag.find("\"", begin)
    return String(tag[byte=begin:end])


def _array_tag(text: String, marker: String) raises -> String:
    """The self-closing DataArray tag containing `marker`, which must
    identify exactly one array."""
    assert_true(text.count(marker) == 1, "expected exactly one array with " + marker)
    var at = text.find(marker)
    var begin = at
    while begin > 0 and String(text[byte=begin]) != "<":
        begin -= 1
    var end = text.find("/>", at)
    assert_true(end > at, "DataArray tag is not self-closing for " + marker)
    var tag = String(text[byte=begin:end])
    assert_true(tag.startswith("<DataArray"), "marker is not inside a DataArray tag")
    assert_true(_attr(tag, "format") == "appended", "array is not appended-binary")
    return tag


def _u64(blob: List[UInt8], at: Int) -> UInt64:
    var u: UInt64 = 0
    for k in range(8):
        u |= UInt64(blob[at + k]) << UInt64(8 * k)
    return u


def _block(vtu: VtuFile, marker: String, dtype: String, elem_bytes: Int) raises -> Tuple[Int, Int]:
    """(start, count) of the payload of the array identified by `marker`,
    after checking its declared type and block header."""
    var tag = _array_tag(vtu.text, marker)
    assert_true(_attr(tag, "type") == dtype, marker + ": wrong DataArray type")
    var offset = atol(_attr(tag, "offset"))
    assert_true(offset >= 0 and offset + 8 <= len(vtu.blob), marker + ": offset out of range")
    var nbytes = Int(_u64(vtu.blob, offset))
    assert_true(nbytes % elem_bytes == 0 and offset + 8 + nbytes <= len(vtu.blob),
                marker + ": block size inconsistent with the file")
    return (offset + 8, nbytes // elem_bytes)


def _floats(vtu: VtuFile, marker: String) raises -> List[Real]:
    var blk = _block(vtu, marker, "Float64", 8)
    var vals = List[Real](capacity=blk[1])
    for i in range(blk[1]):
        vals.append(bitcast[DType.float64, 1](_u64(vtu.blob, blk[0] + 8 * i)))
    return vals^


def _ints(vtu: VtuFile, marker: String, dtype: String) raises -> List[Int]:
    var width = 4 if dtype == "Int32" else 1
    var blk = _block(vtu, marker, dtype, width)
    var vals = List[Int](capacity=blk[1])
    for i in range(blk[1]):
        var v = 0
        for k in range(width):
            v |= Int(vtu.blob[blk[0] + width * i + k]) << (8 * k)
        vals.append(v)
    return vals^


def _attr_int(text: String, key: String) raises -> Int:
    """Integer value of the single `key="N"` attribute."""
    return atol(_attr(text, key))


# --- checks ---------------------------------------------------------------

def _check_grid(
    label: String, vtu: VtuFile, ndim: Int, order: Int, num_elements: Int,
    coords: List[Real], measure: Real, num_fields: Int,
) raises:
    """Structure, geometry and topology of one VTU file."""
    var text = vtu.text.copy()
    var nodes_per_elem = (order + 1) ** ndim
    var cells_per_elem = order ** ndim
    var num_points = num_elements * nodes_per_elem
    var num_cells = num_elements * cells_per_elem
    var corners = 4 if ndim == 2 else 8
    var cell_type = 9 if ndim == 2 else 12

    assert_true(text.startswith("<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\""),
                label + ": bad header")
    assert_true("byte_order=\"LittleEndian\"" in text and "header_type=\"UInt64\"" in text,
                label + ": header does not declare the binary encoding")
    assert_true(text.endswith("</AppendedData>\n</VTKFile>\n"), label + ": bad trailer")
    for tag in ["VTKFile", "UnstructuredGrid", "Piece", "Points", "Cells", "PointData",
                "AppendedData"]:
        assert_true(text.count("<" + tag) == 1 and text.count("</" + tag + ">") == 1,
                    label + ": unbalanced <" + tag + ">")
    # points + connectivity + offsets + types + the point-data arrays.
    assert_true(text.count("<DataArray") == 4 + num_fields
                and text.count("</DataArray>") == 0,
                label + ": wrong number of DataArray elements")
    # The blocks tile the appended section exactly: nothing missing, no
    # trailing bytes. Each block is an 8-byte header plus its payload.
    var expected_bytes = (
        (8 + num_points * 24) + (8 + num_cells * corners * 4)
        + (8 + num_cells * 4) + (8 + num_cells)
        + num_fields * (8 + num_points * 8)
    )
    assert_true(len(vtu.blob) == expected_bytes,
                label + ": appended-data size does not match the declared arrays")

    assert_true(_attr_int(text, "NumberOfPoints") == num_points,
                label + ": declared NumberOfPoints != mesh node count")
    assert_true(_attr_int(text, "NumberOfCells") == num_cells,
                label + ": declared NumberOfCells != mesh sub-cell count")

    # Points: one xyz triple per DG node, equal to the node coordinates.
    var pts = _floats(vtu, "NumberOfComponents=\"3\"")
    assert_true(len(coords) == 3 * num_points, label + ": test coordinate list has wrong length")
    assert_true(len(pts) == 3 * num_points, label + ": point value count != 3 * NumberOfPoints")
    var max_dx: Real = 0.0
    for i in range(len(pts)):
        var d = _abs(pts[i] - coords[i])
        if d > max_dx: max_dx = d
    assert_true(max_dx < COORD_TOL, label + ": point coordinates differ from DG nodes")

    # Connectivity.
    var conn = _ints(vtu, "Name=\"connectivity\"", "Int32")
    assert_true(len(conn) == corners * num_cells,
                label + ": connectivity count != corners * NumberOfCells")
    var total: Real = 0.0
    for c in range(num_cells):
        var elem = c // cells_per_elem
        for k in range(corners):
            var p = conn[c * corners + k]
            assert_true(p >= elem * nodes_per_elem and p < (elem + 1) * nodes_per_elem,
                        label + ": sub-cell references a node outside its element")
        var p0 = 3 * conn[c * corners + 0]
        var p1 = 3 * conn[c * corners + 1]
        var p2 = 3 * conn[c * corners + 2]
        var p3 = 3 * conn[c * corners + 3]
        # Edge vectors of the axis-aligned brick, from the VTK corner order.
        var ex = pts[p1] - pts[p0]
        var ey = pts[p3 + 1] - pts[p0 + 1]
        assert_true(ex > Real(0.0) and ey > Real(0.0), label + ": sub-cell is inverted")
        # Corner layout of the bottom face: 0 → +x → +x+y → +y.
        var bad = (
            _abs(pts[p1 + 1] - pts[p0 + 1]) + _abs(pts[p3] - pts[p0])
            + _abs(pts[p2] - pts[p1]) + _abs(pts[p2 + 1] - pts[p3 + 1])
        )
        var size = ex * ey
        if ndim == 3:
            var ez = pts[3 * conn[c * corners + 4] + 2] - pts[p0 + 2]
            assert_true(ez > Real(0.0), label + ": sub-cell is inverted in z")
            for k in range(4):
                var lo = 3 * conn[c * corners + k]
                var hi = 3 * conn[c * corners + 4 + k]
                # Top face sits directly above the bottom face.
                bad += (
                    _abs(pts[hi] - pts[lo]) + _abs(pts[hi + 1] - pts[lo + 1])
                    + _abs(pts[hi + 2] - pts[lo + 2] - ez)
                    + _abs(pts[lo + 2] - pts[p0 + 2])
                )
            size *= ez
        assert_true(bad < Real(1.0e-13), label + ": sub-cell corners are not in VTK order")
        total += size
    assert_true(_abs(total - measure) < Real(1.0e-12) * measure,
                label + ": sub-cells do not tile the domain")

    var offsets = _ints(vtu, "Name=\"offsets\"", "Int32")
    var types = _ints(vtu, "Name=\"types\"", "UInt8")
    assert_true(len(offsets) == num_cells and len(types) == num_cells,
                label + ": offsets / types count != NumberOfCells")
    for c in range(num_cells):
        assert_true(offsets[c] == corners * (c + 1), label + ": wrong cell offset")
        assert_true(types[c] == cell_type, label + ": wrong VTK cell type")


def _check_fields(
    label: String, vtu: VtuFile, names: List[String], coords: List[Real],
    shift: Real,
) raises -> Real:
    """Every named point-data array equals the analytic field at its
    point. Returns the largest deviation seen."""
    var num_points = len(coords) // 3
    var worst: Real = 0.0
    for name in names:
        var vals = _floats(vtu, "Name=\"" + name + "\"")
        assert_true(len(vals) == num_points,
                    label + ": " + name + " value count != NumberOfPoints")
        for i in range(num_points):
            var want = _field(
                name, coords[3 * i], coords[3 * i + 1], coords[3 * i + 2], shift,
            )
            var d = _abs(vals[i] - want)
            if d > worst: worst = d
            assert_true(d < VALUE_TOL, label + ": " + name + " differs from the state")
    return worst


def _check(
    label: String, path: String, ndim: Int, order: Int,
    names: List[String], coords: List[Real], shift: Real,
) raises:
    var vtu = _read(path)
    var num_elements = NX * NY if ndim == 2 else NX * NY * NZ
    var measure = (X_MAX - X_MIN) * (Y_MAX - Y_MIN)
    if ndim == 3:
        measure *= Z_MAX - Z_MIN
    _check_grid(label, vtu, ndim, order, num_elements, coords, measure, len(names))
    var worst = _check_fields(label, vtu, names, coords, shift)
    print(label, ":", len(coords) // 3, "points,", len(names), "fields, max |Δ| =", worst)
    remove(path)


def _mesh_2d[P: Int](var ctx: DeviceContext) raises -> BlockMesh[2, P]:
    return BlockMesh[2, P](
        ctx, X_MIN, X_MAX, Y_MIN, Y_MAX,
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )


def _mesh_3d[P: Int](var ctx: DeviceContext) raises -> BlockMesh[3, P]:
    return BlockMesh[3, P](
        ctx, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, nx=NX, ny=NY, nz=NZ,
    )


def _test_2d[P: Int](var ctx: DeviceContext) raises:
    var mesh = _mesh_2d[P](ctx)

    var q_euler = DGState[2, P, EULER_NC](ctx, mesh.num_elements)
    var xy = _fill_2d[P, EULER_NC](ctx, q_euler, KIND_FLUID, Real(0.0))
    var path = _fresh(OUTPUT_DIR + "/euler_2d.vtu")
    dump_euler_frame(path, mesh, q_euler, GAMMA)
    _check("Euler 2D", path, 2, P, ["rho", "u", "v", "w", "P"], xy, Real(0.0))

    var q_mhd = DGState[2, P, MHD_NC](ctx, mesh.num_elements)
    xy = _fill_2d[P, MHD_NC](ctx, q_mhd, KIND_FLUID, Real(0.0))
    path = _fresh(OUTPUT_DIR + "/imhd_2d.vtu")
    dump_imhd_frame(path, mesh, q_mhd, GAMMA)
    _check("IMHD 2D", path, 2, P,
           ["rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2"], xy, Real(0.0))

    var q_glm = DGState[2, P, MHD_GLM_NC](ctx, mesh.num_elements)
    xy = _fill_2d[P, MHD_GLM_NC](ctx, q_glm, KIND_FLUID, Real(0.0))
    path = _fresh(OUTPUT_DIR + "/imhd_glm_2d.vtu")
    dump_imhd_glm_frame(path, mesh, q_glm, GAMMA)
    _check("GLM-MHD 2D", path, 2, P,
           ["rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2", "psi"], xy, Real(0.0))

    var q_max = DGState[2, P, MAXWELL_NC](ctx, mesh.num_elements)
    xy = _fill_2d[P, MAXWELL_NC](ctx, q_max, KIND_MAXWELL, Real(0.0))
    path = _fresh(OUTPUT_DIR + "/maxwell_2d.vtu")
    dump_maxwell_frame(path, mesh, q_max)
    _check("Maxwell 2D", path, 2, P,
           ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z", "E_energy", "B_energy"],
           xy, Real(0.0))


def _test_3d[P: Int](var ctx: DeviceContext) raises:
    var mesh = _mesh_3d[P](ctx)

    var q_euler = DGState[3, P, EULER_NC](ctx, mesh.num_elements)
    var xyz = _fill_3d[P, EULER_NC](ctx, q_euler, KIND_FLUID)
    var path = _fresh(OUTPUT_DIR + "/euler_3d.vtu")
    dump_euler_frame(path, mesh, q_euler, GAMMA)
    _check("Euler 3D", path, 3, P, ["rho", "u", "v", "w", "P"], xyz, Real(0.0))

    var q_mhd = DGState[3, P, MHD_NC](ctx, mesh.num_elements)
    xyz = _fill_3d[P, MHD_NC](ctx, q_mhd, KIND_FLUID)
    path = _fresh(OUTPUT_DIR + "/imhd_3d.vtu")
    dump_imhd_frame(path, mesh, q_mhd, GAMMA)
    _check("IMHD 3D", path, 3, P,
           ["rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2"], xyz, Real(0.0))

    var q_glm = DGState[3, P, MHD_GLM_NC](ctx, mesh.num_elements)
    xyz = _fill_3d[P, MHD_GLM_NC](ctx, q_glm, KIND_FLUID)
    path = _fresh(OUTPUT_DIR + "/imhd_glm_3d.vtu")
    dump_imhd_glm_frame(path, mesh, q_glm, GAMMA)
    _check("GLM-MHD 3D", path, 3, P,
           ["rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2", "psi"],
           xyz, Real(0.0))

    var q_max = DGState[3, P, MAXWELL_NC](ctx, mesh.num_elements)
    xyz = _fill_3d[P, MAXWELL_NC](ctx, q_max, KIND_MAXWELL)
    path = _fresh(OUTPUT_DIR + "/maxwell_3d.vtu")
    dump_maxwell_frame(path, mesh, q_max)
    _check("Maxwell 3D", path, 3, P,
           ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z", "E_energy", "B_energy"],
           xyz, Real(0.0))


def _test_frames[P: Int](var ctx: DeviceContext) raises:
    """Three frames with different states and times, then the PVD."""
    var mesh = _mesh_2d[P](ctx)
    var q = DGState[2, P, MHD_GLM_NC](ctx, mesh.num_elements)
    var times = [Real(0.0), Real(0.125), Real(2.5)]
    var shifts = [Real(0.0), Real(0.5), Real(1.25)]
    var names: List[String] = [
        "rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2", "psi",
    ]

    var pvd_path = _fresh(OUTPUT_DIR + "/frames.pvd")
    for i in range(3):
        _ = _fresh(OUTPUT_DIR + "/frames_000" + String(i) + ".vtu")

    var fw = FrameWriter("frames", OUTPUT_DIR)
    var coords = List[List[Real]]()
    for i in range(3):
        coords.append(_fill_2d[P, MHD_GLM_NC](ctx, q, KIND_FLUID, shifts[i]))
        var path = fw.next_path(times[i])
        assert_true(path == OUTPUT_DIR + "/frames_000" + String(i) + ".vtu",
                    "FrameWriter returned an unexpected path: " + path)
        dump_imhd_glm_frame(path, mesh, q, GAMMA)
    fw.write_pvd()

    # PVD: one DataSet per frame, in order, each pairing the right time
    # with the right (relative) file name, and the file must exist.
    var pvd = _read_text(pvd_path)
    assert_true(pvd.startswith("<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\""),
                "PVD: bad header")
    assert_true(pvd.endswith("  </Collection>\n</VTKFile>\n"), "PVD: bad trailer")
    assert_true(pvd.count("<Collection>") == 1 and pvd.count("</Collection>") == 1,
                "PVD: unbalanced Collection")
    var datasets = List[String]()
    for line in pvd.split("\n"):
        if "<DataSet" in line:
            datasets.append(String(line))
    assert_true(len(datasets) == 3, "PVD does not list exactly three frames")
    for i in range(3):
        var rel = "frames_000" + String(i) + ".vtu"
        assert_true("timestep=\"" + String(times[i]) + "\"" in datasets[i],
                    "PVD frame " + String(i) + " has the wrong timestep")
        assert_true("file=\"" + rel + "\"" in datasets[i],
                    "PVD frame " + String(i) + " points at the wrong file")
        assert_true(datasets[i].endswith("/>"), "PVD DataSet element not closed")
        # Each frame holds its own state, not a copy of another frame.
        _check("frame " + String(i), OUTPUT_DIR + "/" + rel, 2, P, names,
               coords[i], shifts[i])
    remove(pvd_path)

    # Frame numbering past one digit (no files written).
    var fw2 = FrameWriter("pad", OUTPUT_DIR)
    var last = String("")
    for i in range(101):
        last = fw2.next_path(Real(i))
        if i == 9:
            assert_true(last == OUTPUT_DIR + "/pad_0009.vtu", "bad frame name at 9")
        if i == 10:
            assert_true(last == OUTPUT_DIR + "/pad_0010.vtu", "bad frame name at 10")
    assert_true(last == OUTPUT_DIR + "/pad_0100.vtu", "bad frame name at 100")
    assert_true(len(fw2.frame_files) == 101 and fw2.frame_times[100] == Real(100.0),
                "FrameWriter lost a frame record")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    makedirs(OUTPUT_DIR, exist_ok=True)

    # Two orders, so the P-dependent sub-cell connectivity is exercised
    # at more than one size.
    _test_2d[2](ctx)
    _test_2d[3](ctx)
    _test_3d[2](ctx)
    _test_3d[1](ctx)
    _test_frames[2](ctx)

    rmdir(OUTPUT_DIR)
    print("PASSED")
