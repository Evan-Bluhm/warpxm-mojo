# Time-series VTU frame writer + PVD collection.
#
# Usage:
#
#     var fw = FrameWriter("orszag_tang", "/tmp/ot")
#     dump_imhd_2d_frame(fw.next_path(0.0), ctx, mesh, q, gamma)
#     for ...:
#         stepper.step(...)
#         if step % N == 0:
#             dump_imhd_2d_frame(fw.next_path(t), ctx, mesh, q, gamma)
#     fw.write_pvd()
#
# `next_path(t)` records the (filename, time) pair and returns the full
# path to write to. `write_pvd()` finalises a ParaView Collection (.pvd)
# file pointing at each frame with its simulation time, so ParaView's
# time-slider lets the user scrub through the run.

from std.collections import List

from warpxm.core.types import Real


struct FrameWriter:
    """Tracks per-frame filenames + simulation times and writes the PVD file."""

    var basename: String
    """File-naming stem, e.g. `"orszag_tang"`."""

    var directory: String
    """Directory all frames + the PVD land in."""

    var frame_files: List[String]
    """Relative frame filenames, e.g. `"orszag_tang_0003.vtu"`."""

    var frame_times: List[Real]

    def __init__(out self, basename: String, directory: String) raises:
        """`directory` must already exist — Mojo's stdlib doesn't yet expose
        a portable `mkdir`. Callers can `mkdir -p` ahead of time."""
        self.basename = basename
        self.directory = directory
        self.frame_files = List[String]()
        self.frame_times = List[Real]()

    def _pad4(self, idx: Int) -> String:
        """Zero-pad an integer to four digits."""
        if idx < 10:
            return "000" + String(idx)
        if idx < 100:
            return "00" + String(idx)
        if idx < 1000:
            return "0" + String(idx)
        return String(idx)

    def next_path(mut self, time: Real) raises -> String:
        """Reserve a new frame filename, record the time, return full path."""
        var idx = len(self.frame_files)
        var rel = self.basename + "_" + self._pad4(idx) + ".vtu"
        self.frame_files.append(rel)
        self.frame_times.append(time)
        return self.directory + "/" + rel

    def write_pvd(self) raises:
        """Emit `<directory>/<basename>.pvd` pointing at every frame."""
        var pvd_path = self.directory + "/" + self.basename + ".pvd"
        var out = open(pvd_path, "w")
        out.write("<?xml version=\"1.0\"?>\n")
        out.write("<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">\n")
        out.write("  <Collection>\n")
        for i in range(len(self.frame_files)):
            out.write("    <DataSet timestep=\"")
            out.write(String(self.frame_times[i]))
            out.write("\" group=\"\" part=\"0\" file=\"")
            out.write(self.frame_files[i])
            out.write("\"/>\n")
        out.write("  </Collection>\n")
        out.write("</VTKFile>\n")
