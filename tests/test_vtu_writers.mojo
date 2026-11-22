# Round-trip test for every VTU frame writer and the PVD frame collection.
#
# Covers `driver/vtu_2d`, `vtu_3d`, `vtu_imhd_2d`, `vtu_imhd_3d` (plain
# and GLM), `vtu_maxwell_2d`, `vtu_maxwell_3d` and `driver/frame_writer`.
#
# Each writer is handed a state built from a known analytic primitive
# field on a mesh with unequal element counts and extents (so a swapped
# axis or stride shows up). The file is then read back and parsed:
#
#   * exactly one well-formed VTKFile / Piece, balanced DataArray tags,
#     and exactly the expected set of point-data arrays;
#   * declared NumberOfPoints / NumberOfCells equal the mesh counts AND
#     the number of values actually present in every array;
#   * point coordinates equal the DG node coordinates, in node order;
#   * connectivity indices are in range, every sub-cell has positive
#     orientation and the right corner layout, and the sub-cell measures
#     sum to the domain area / volume; offsets and cell types are right;
#   * every point-data array equals the analytic value at its point
#     (derived quantities such as u = m/ρ and P included) to 1e-14 — the
#     writer prints shortest round-trip Float64, so only arithmetic
#     round-off is left.
#
# Output goes to a directory owned by this test; any file left by an
# earlier run is deleted before it is rewritten, so a writer that
# silently writes nothing cannot pass on a stale file.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vtu_writers.mojo

from std.collections import List
from std.collections.string import atof, atol
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
from warpxm.driver.frame_writer import FrameWriter
from warpxm.driver.vtu_2d import dump_euler_2d_frame
from warpxm.driver.vtu_3d import dump_euler_3d_frame
from warpxm.driver.vtu_imhd_2d import dump_imhd_2d_frame, dump_imhd_glm_2d_frame
from warpxm.driver.vtu_imhd_3d import dump_imhd_3d_frame, dump_imhd_glm_3d_frame
from warpxm.driver.vtu_maxwell_2d import dump_maxwell_2d_frame
from warpxm.driver.vtu_maxwell_3d import dump_maxwell_3d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_2d import DGState2D
from warpxm.variable.state_3d import DGState3D


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
    var ctx: DeviceContext, mut q: DGState2D[P, NC], kind: Int, shift: Real,
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
    var ctx: DeviceContext, mut q: DGState3D[P, NC], kind: Int,
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


def _read(path: String) raises -> String:
    assert_true(exists(path), "writer produced no file at " + path)
    return String(open(path, "r").read())


def _abs(v: Real) -> Real:
    return v if v >= Real(0.0) else -v


def _array_body(text: String, marker: String) raises -> String:
    """Text between the DataArray open tag containing `marker` and its
    close tag. `marker` must identify exactly one array."""
    assert_true(text.count(marker) == 1, "expected exactly one array with " + marker)
    var at = text.find(marker)
    var begin = text.find(">", at) + 1
    var end = text.find("</DataArray>", begin)
    assert_true(begin > 0 and end > begin, "malformed DataArray for " + marker)
    return String(text[byte=begin:end])


def _floats(body: String) raises -> List[Real]:
    var out = List[Real]()
    for tok in body.split():
        out.append(atof(tok))
    return out^


def _ints(body: String) raises -> List[Int]:
    var out = List[Int]()
    for tok in body.split():
        out.append(atol(tok))
    return out^


def _attr_int(text: String, key: String) raises -> Int:
    """Integer value of the single `key="N"` attribute."""
    var marker = key + "=\""
    assert_true(text.count(marker) == 1, "expected exactly one " + key)
    var begin = text.find(marker) + marker.byte_length()
    var end = text.find("\"", begin)
    return atol(String(text[byte=begin:end]))


# --- checks ---------------------------------------------------------------

def _check_grid(
    label: String, text: String, ndim: Int, order: Int, num_elements: Int,
    coords: List[Real], measure: Real, num_fields: Int,
) raises:
    """Structure, geometry and topology of one VTU file."""
    var nodes_per_elem = (order + 1) ** ndim
    var cells_per_elem = order ** ndim
    var num_points = num_elements * nodes_per_elem
    var num_cells = num_elements * cells_per_elem
    var corners = 4 if ndim == 2 else 8
    var cell_type = 9 if ndim == 2 else 12

    assert_true(text.startswith("<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\""),
                label + ": bad header")
    assert_true(text.endswith("</UnstructuredGrid>\n</VTKFile>\n"), label + ": bad trailer")
    for tag in ["VTKFile", "UnstructuredGrid", "Piece", "Points", "Cells", "PointData"]:
        assert_true(text.count("<" + tag) == 1 and text.count("</" + tag + ">") == 1,
                    label + ": unbalanced <" + tag + ">")
    # points + connectivity + offsets + types + the point-data arrays.
    assert_true(text.count("<DataArray") == 4 + num_fields
                and text.count("</DataArray>") == 4 + num_fields,
                label + ": wrong number of DataArray elements")

    assert_true(_attr_int(text, "NumberOfPoints") == num_points,
                label + ": declared NumberOfPoints != mesh node count")
    assert_true(_attr_int(text, "NumberOfCells") == num_cells,
                label + ": declared NumberOfCells != mesh sub-cell count")

    # Points: one xyz triple per DG node, equal to the node coordinates.
    var pts = _floats(_array_body(text, "NumberOfComponents=\"3\""))
    assert_true(len(coords) == 3 * num_points, label + ": test coordinate list has wrong length")
    assert_true(len(pts) == 3 * num_points, label + ": point value count != 3 * NumberOfPoints")
    var max_dx: Real = 0.0
    for i in range(len(pts)):
        var d = _abs(pts[i] - coords[i])
        if d > max_dx: max_dx = d
    assert_true(max_dx < COORD_TOL, label + ": point coordinates differ from DG nodes")

    # Connectivity.
    var conn = _ints(_array_body(text, "Name=\"connectivity\""))
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

    var offsets = _ints(_array_body(text, "Name=\"offsets\""))
    var types = _ints(_array_body(text, "Name=\"types\""))
    assert_true(len(offsets) == num_cells and len(types) == num_cells,
                label + ": offsets / types count != NumberOfCells")
    for c in range(num_cells):
        assert_true(offsets[c] == corners * (c + 1), label + ": wrong cell offset")
        assert_true(types[c] == cell_type, label + ": wrong VTK cell type")


def _check_fields(
    label: String, text: String, names: List[String], coords: List[Real],
    shift: Real,
) raises -> Real:
    """Every named point-data array equals the analytic field at its
    point. Returns the largest deviation seen."""
    var num_points = len(coords) // 3
    var worst: Real = 0.0
    for name in names:
        var vals = _floats(_array_body(text, "Name=\"" + name + "\""))
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
    var text = _read(path)
    var num_elements = NX * NY if ndim == 2 else NX * NY * NZ
    var measure = (X_MAX - X_MIN) * (Y_MAX - Y_MIN)
    if ndim == 3:
        measure *= Z_MAX - Z_MIN
    _check_grid(label, text, ndim, order, num_elements, coords, measure, len(names))
    var worst = _check_fields(label, text, names, coords, shift)
    print(label, ":", len(coords) // 3, "points,", len(names), "fields, max |Δ| =", worst)
    remove(path)


def _mesh_2d[P: Int](var ctx: DeviceContext) raises -> BlockMesh2D[P]:
    return BlockMesh2D[P](
        ctx, X_MIN, X_MAX, Y_MIN, Y_MAX,
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )


def _mesh_3d[P: Int](var ctx: DeviceContext) raises -> BlockMesh3D[P]:
    return BlockMesh3D[P](
        ctx, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, nx=NX, ny=NY, nz=NZ,
    )


def _test_2d[P: Int](var ctx: DeviceContext) raises:
    var mesh = _mesh_2d[P](ctx)

    var q_euler = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
    var xy = _fill_2d[P, EULER_NC](ctx, q_euler, KIND_FLUID, Real(0.0))
    var path = _fresh(OUTPUT_DIR + "/euler_2d.vtu")
    dump_euler_2d_frame[P](path, ctx, mesh, q_euler, GAMMA)
    _check("Euler 2D", path, 2, P, ["rho", "u", "v", "P"], xy, Real(0.0))

    var q_mhd = DGState2D[P, MHD_NC](ctx, mesh.num_elements)
    xy = _fill_2d[P, MHD_NC](ctx, q_mhd, KIND_FLUID, Real(0.0))
    path = _fresh(OUTPUT_DIR + "/imhd_2d.vtu")
    dump_imhd_2d_frame[P](path, ctx, mesh, q_mhd, GAMMA)
    _check("IMHD 2D", path, 2, P,
           ["rho", "u", "v", "P", "Bx", "By", "B2_over_2"], xy, Real(0.0))

    var q_glm = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    xy = _fill_2d[P, MHD_GLM_NC](ctx, q_glm, KIND_FLUID, Real(0.0))
    path = _fresh(OUTPUT_DIR + "/imhd_glm_2d.vtu")
    dump_imhd_glm_2d_frame[P, MHD_GLM_NC](path, ctx, mesh, q_glm, GAMMA)
    _check("GLM-MHD 2D", path, 2, P,
           ["rho", "u", "v", "P", "Bx", "By", "B2_over_2", "psi"], xy, Real(0.0))

    var q_max = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)
    xy = _fill_2d[P, MAXWELL_NC](ctx, q_max, KIND_MAXWELL, Real(0.0))
    path = _fresh(OUTPUT_DIR + "/maxwell_2d.vtu")
    dump_maxwell_2d_frame[P](path, ctx, mesh, q_max)
    _check("Maxwell 2D", path, 2, P,
           ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z", "E_energy", "B_energy"],
           xy, Real(0.0))


def _test_3d[P: Int](var ctx: DeviceContext) raises:
    var mesh = _mesh_3d[P](ctx)

    var q_euler = DGState3D[P, EULER_NC](ctx, mesh.num_elements)
    var xyz = _fill_3d[P, EULER_NC](ctx, q_euler, KIND_FLUID)
    var path = _fresh(OUTPUT_DIR + "/euler_3d.vtu")
    dump_euler_3d_frame[P](path, ctx, mesh, q_euler, GAMMA)
    _check("Euler 3D", path, 3, P, ["rho", "u", "v", "w", "P"], xyz, Real(0.0))

    var q_mhd = DGState3D[P, MHD_NC](ctx, mesh.num_elements)
    xyz = _fill_3d[P, MHD_NC](ctx, q_mhd, KIND_FLUID)
    path = _fresh(OUTPUT_DIR + "/imhd_3d.vtu")
    dump_imhd_3d_frame[P](path, ctx, mesh, q_mhd, GAMMA)
    _check("IMHD 3D", path, 3, P,
           ["rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2"], xyz, Real(0.0))

    var q_glm = DGState3D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    xyz = _fill_3d[P, MHD_GLM_NC](ctx, q_glm, KIND_FLUID)
    path = _fresh(OUTPUT_DIR + "/imhd_glm_3d.vtu")
    dump_imhd_glm_3d_frame[P, MHD_GLM_NC](path, ctx, mesh, q_glm, GAMMA)
    _check("GLM-MHD 3D", path, 3, P,
           ["rho", "u", "v", "w", "P", "Bx", "By", "Bz", "B2_over_2", "psi"],
           xyz, Real(0.0))

    var q_max = DGState3D[P, MAXWELL_NC](ctx, mesh.num_elements)
    xyz = _fill_3d[P, MAXWELL_NC](ctx, q_max, KIND_MAXWELL)
    path = _fresh(OUTPUT_DIR + "/maxwell_3d.vtu")
    dump_maxwell_3d_frame[P](path, ctx, mesh, q_max)
    _check("Maxwell 3D", path, 3, P,
           ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z", "E_energy", "B_energy"],
           xyz, Real(0.0))


def _test_frames[P: Int](var ctx: DeviceContext) raises:
    """Three frames with different states and times, then the PVD."""
    var mesh = _mesh_2d[P](ctx)
    var q = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var times = [Real(0.0), Real(0.125), Real(2.5)]
    var shifts = [Real(0.0), Real(0.5), Real(1.25)]
    var names: List[String] = ["rho", "u", "v", "P", "Bx", "By", "B2_over_2", "psi"]

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
        dump_imhd_glm_2d_frame[P, MHD_GLM_NC](path, ctx, mesh, q, GAMMA)
    fw.write_pvd()

    # PVD: one DataSet per frame, in order, each pairing the right time
    # with the right (relative) file name, and the file must exist.
    var pvd = _read(pvd_path)
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
