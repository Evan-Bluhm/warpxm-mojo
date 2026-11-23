# Cartesian domain decomposition of a block mesh.
#
# The global `n[0] × n[1] × n[2]` element grid is cut into a
# `procs[0] × procs[1] × procs[2]` grid of boxes, one per rank. Rank
# `(rz · procs[1] + ry) · procs[0] + rx` owns box `(rx, ry, rz)`. The
# process grid is the factorisation of the rank count with the smallest
# total interface area, so the halo traffic per step is minimal.
#
# Everything here is host arithmetic on (rank, size): every rank computes
# the same decomposition without communicating.

from std.collections import Array


@fieldwise_init
struct Partition(Copyable, Movable):
    """One rank's box of the global element grid."""

    var global_n: Array[Int, 3]
    """Elements of the whole mesh along each axis."""

    var procs: Array[Int, 3]
    """Boxes along each axis."""

    var coords: Array[Int, 3]
    """This rank's box."""

    var start: Array[Int, 3]
    """Global index of the first owned element along each axis."""

    var n: Array[Int, 3]
    """Owned elements along each axis."""

    var neighbor: Array[Int, 6]
    """Rank across side `2·axis + side` (side 0 negative, 1 positive), or
    -1 where there is none to exchange with: a physical boundary, or an
    axis with a single box (a periodic axis then wraps onto the rank
    itself, as on a serial mesh)."""

    def global_element(self, e: Int) -> Int:
        """Index in the whole mesh of this rank's element `e`."""
        var i = self.start[0] + e % self.n[0]
        var j = self.start[1] + (e // self.n[0]) % self.n[1]
        var k = self.start[2] + e // (self.n[0] * self.n[1])
        return (k * self.global_n[1] + j) * self.global_n[0] + i

    def is_split(self) -> Bool:
        """True when more than one rank shares the mesh."""
        return self.procs[0] * self.procs[1] * self.procs[2] > 1


def _interface_area(n: Array[Int, 3], p: Array[Int, 3]) -> Int:
    """Faces exchanged per step, up to a constant: an axis cut into
    `p > 1` boxes contributes `p` interfaces of the transverse area."""
    var total = n[0] * n[1] * n[2]
    var area = 0
    for a in range(3):
        if p[a] > 1:
            area += p[a] * (total // n[a])
    return area


def process_grid(n: Array[Int, 3], ndim: Int, size: Int) raises -> Array[Int, 3]:
    """The factorisation `size = p0 · p1 · p2` with the least interface
    area, using only the first `ndim` axes and at most `n[a]` boxes along
    axis `a`. Ties go to cutting the higher axis, which keeps each box
    contiguous in the element numbering."""
    var best: Array[Int, 3] = [0, 0, 0]
    var best_area = -1
    for p0 in range(1, size + 1):
        if size % p0 != 0 or p0 > n[0]:
            continue
        var rest = size // p0
        for p1 in range(1, rest + 1):
            if rest % p1 != 0:
                continue
            var p: Array[Int, 3] = [p0, p1, rest // p1]
            var ok = True
            for a in range(3):
                if p[a] > n[a] or (a >= ndim and p[a] != 1):
                    ok = False
            if not ok:
                continue
            var area = _interface_area(n, p)
            if best_area < 0 or area < best_area:
                best_area = area
                best = p.copy()
    if best_area < 0:
        raise Error(
            "cannot split a " + String(n[0]) + " x " + String(n[1]) + " x "
            + String(n[2]) + " element mesh over " + String(size) + " ranks"
        )
    return best^


def decompose(
    n: Array[Int, 3], periodic: Array[Bool, 3], ndim: Int,
    rank: Int, size: Int,
) raises -> Partition:
    """The box of `rank` when the mesh is shared by `size` ranks."""
    var procs = process_grid(n, ndim, size)
    var coords: Array[Int, 3] = [
        rank % procs[0],
        (rank // procs[0]) % procs[1],
        rank // (procs[0] * procs[1]),
    ]
    var start: Array[Int, 3] = [0, 0, 0]
    var local_n: Array[Int, 3] = [1, 1, 1]
    var neighbor = Array[Int, 6](fill=-1)
    for a in range(3):
        start[a] = (n[a] * coords[a]) // procs[a]
        local_n[a] = (n[a] * (coords[a] + 1)) // procs[a] - start[a]
        if procs[a] == 1:
            continue
        for side in range(2):
            var c = coords.copy()
            c[a] = coords[a] + (1 if side == 1 else -1)
            if c[a] < 0 or c[a] >= procs[a]:
                if not periodic[a]:
                    continue
                c[a] = (c[a] + procs[a]) % procs[a]
            neighbor[2 * a + side] = (c[2] * procs[1] + c[1]) * procs[0] + c[0]
    return Partition(n.copy(), procs^, coords^, start^, local_n^, neighbor^)
