# Multi-rank regression: VTU frame output.
#
# Every rank writes its own piece of each frame; rank 0 adds the `.pvtu`
# index of the frame and the `.pvd` time series. This checks that the set
# of files is complete and consistent: one piece per rank, the pieces'
# point counts add up to the whole mesh, and the series points at the file
# to open (the `.pvtu` on several ranks, the `.vtu` itself on one).

from std.collections import Array
from std.math import sin
from std.os.path import exists
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *
from warpxm.parallel import mpi

comptime P: Int = 2
comptime NX: Int = 9
comptime NY: Int = 7
comptime State = DGState[2, P, EULER_NC]


def _count(text: String, needle: String) -> Int:
    return len(text.split(needle)) - 1


def _points_in(path: String) raises -> Int:
    """`NumberOfPoints` of the piece in a `.vtu` file."""
    # The file ends in raw binary; the attribute is in the XML header.
    var bytes = open(path, "r").read_bytes()
    var text = String(from_utf8_lossy=Span[UInt8, _](bytes)[0:400])
    var after = text.split("NumberOfPoints=\"")[1]
    return Int(String(after.split("\"")[0]))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)
    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        return [
            Real(1.0) + Real(0.1) * sin(TWO_PI * x) * sin(TWO_PI * y),
            Real(0.0), Real(0.0), Real(0.0), Real(2.5),
        ]
    mesh.fill(q, initial)

    var fw = FrameWriter("frames", "vtu_test")
    dump_euler_frame(fw.next_path(Real(0.0)), mesh, q, Real(1.4))
    dump_euler_frame(fw.next_path(Real(0.5)), mesh, q, Real(1.4))
    fw.write_pvd()
    mpi.barrier()   # every rank's files are on disk past this point

    if is_root():
        var ranks = num_ranks()
        var pvd = open("vtu_test/frames.pvd", "r").read()
        if _count(pvd, "<DataSet") != 2:
            raise Error("the .pvd should list two frames")
        var total_points = NX * NY * (P + 1) * (P + 1)
        for frame in ["frames_0000", "frames_0001"]:
            var points = 0
            if ranks == 1:
                if _count(pvd, frame + ".vtu") != 1:
                    raise Error("the .pvd should point at " + frame + ".vtu")
                points = _points_in("vtu_test/" + frame + ".vtu")
            else:
                if _count(pvd, frame + ".pvtu") != 1:
                    raise Error("the .pvd should point at " + frame + ".pvtu")
                var index = open("vtu_test/" + frame + ".pvtu", "r").read()
                if _count(index, "<Piece") != ranks:
                    raise Error("the .pvtu should list one piece per rank")
                for r in range(ranks):
                    var piece = frame + "_p" + String(r) + ".vtu"
                    if _count(index, "\"" + piece + "\"") != 1:
                        raise Error("the .pvtu does not list " + piece)
                    if not exists("vtu_test/" + piece):
                        raise Error("missing piece file " + piece)
                    points += _points_in("vtu_test/" + piece)
            if points != total_points:
                raise Error(
                    frame + ": pieces hold " + String(points)
                    + " points, the mesh has " + String(total_points)
                )
        print("ranks =", ranks, " frames, pieces and series are consistent")
        print("PASSED")
