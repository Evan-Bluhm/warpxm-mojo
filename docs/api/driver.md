# `warpxm.driver`

Output drivers — VTU per-frame writers and the `FrameWriter` helper for
time-series.

## VTU writers

One writer per (physics, dimension):

Field names below are the exact ParaView `DataArray Name=` strings
written by each writer (case-sensitive — type them in verbatim
when colouring).

| Module | Function | Fields |
| ------ | -------- | ------ |
| `driver/vtu_2d.mojo` | `dump_euler_2d_frame[P]` | `rho`, `u`, `v`, `P` |
| `driver/vtu_3d.mojo` | `dump_euler_3d_frame[P]` | `rho`, `u`, `v`, `w`, `P` |
| `driver/vtu_imhd_2d.mojo` | `dump_imhd_2d_frame[P]` (NC=8) / `dump_imhd_glm_2d_frame[P, NC]` (NC=9) | `rho`, `u`, `v`, `P`, `Bx`, `By`, `B2_over_2`, `psi` (NC=9 only) |
| `driver/vtu_imhd_3d.mojo` | `dump_imhd_3d_frame[P]` / `dump_imhd_glm_3d_frame[P, NC]` | as 2D + `w`, `Bz` |
| `driver/vtu_maxwell_2d.mojo` | `dump_maxwell_2d_frame[P]` | `E_x`, `E_y`, `E_z`, `B_x`, `B_y`, `B_z`, `E_energy`, `B_energy` |
| `driver/vtu_maxwell_3d.mojo` | `dump_maxwell_3d_frame[P]` | same as 2D |

Each writer:

* opens `path` for writing as XML VTK UnstructuredGrid,
* writes the geometry (P²-sub-quad or P³-sub-hex decomposition of each
  element so the VTU is order-agnostic),
* writes one `<DataArray>` per field, named as in the tables above.

Element boundaries show the DG discontinuity directly — no averaging.

Usage:

```mojo
from warpxm.driver.vtu_2d import dump_euler_2d_frame
dump_euler_2d_frame[P](
    "/tmp/sod_2d/sod_2d_0010.vtu",
    ctx, mesh, q, gamma,
)
```

## `FrameWriter`

```mojo
from warpxm.driver.frame_writer import FrameWriter

var fw = FrameWriter("sod_2d", "/tmp/sod_2d")

# Inside a time loop:
for frame in range(NUM_FRAMES):
    ctx.synchronize()
    dump_euler_2d_frame[P](fw.next_path(t), ctx, mesh, q, gamma)
    # ... advance time ...

# After the loop:
fw.write_pvd()
```

* `fw.next_path(t)` returns a filename like `<dir>/<basename>_0000.vtu`
  and records the time `t` for the PVD collection.
* `fw.write_pvd()` writes a `.pvd` index referencing every frame with
  its time stamp. Open that in ParaView for an automatic time slider.

## VTU writer mesh decomposition

For a $P+1$-order DG element with $(P+1)^d$ GLL nodes, the writer
decomposes it into $P^d$ linear sub-cells:

| Dim | Sub-cell | VTK cell type |
| --- | -------- | ------------- |
| 2   | quad     | `VTK_QUAD = 9` |
| 3   | hex      | `VTK_HEXAHEDRON = 12` |

The sub-cell connectivity uses each element's GLL nodes directly — no
extrapolation, no averaging. This is what makes the writer **order-
agnostic**: it works at $P = 1$ all the way through $P = 4$ unchanged.

## ParaView usage

Open the `.pvd`, hit Apply, colour by field. See
[Viewing output in ParaView](../getting-started/paraview.md).
