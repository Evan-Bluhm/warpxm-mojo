# `warpxm.io`

Output: one generic ParaView `.vtu` writer, per-physics presets on top of
it, and the `FrameWriter` helper for time series.

## `write_vtu`

```mojo
from warpxm.io.vtu import write_vtu

def fields(v: Array[Real, EULER_NC]) -> Array[Real, 2]:
    return [v[IDX_RHO], v[IDX_MX] / v[IDX_RHO]]
write_vtu(path, mesh, q, ["rho", "u"], fields)
```

`mesh` is a `BlockMesh[2, P]` or `BlockMesh[3, P]` and `q` the matching
`DGState`. `fields` maps one node state to `NF` values; `names` gives the
point-data array name for each, in the same order (a `List[String]` of
length `NF`). The first name becomes the file's default scalar. There is
no 1D writer.

The writer takes no device context: it gathers the state through
`mesh.visit_nodes`, which copies it to the host.

## Presets

```mojo
from warpxm.io.vtu_presets import (
    dump_euler_frame, dump_imhd_frame, dump_imhd_glm_frame,
    dump_maxwell_frame,
)

dump_euler_frame(path, mesh, q, gamma)
dump_maxwell_frame(path, mesh, q)
```

Each preset is overloaded on the mesh dimension and writes the same
arrays in 2D and 3D. The names are the exact ParaView array names
(case-sensitive).

| Function | State | Arrays |
| -------- | ----- | ------ |
| `dump_euler_frame(path, mesh, q, gamma)` | `EULER_NC` | `rho`, `u`, `v`, `w`, `P` |
| `dump_imhd_frame(path, mesh, q, gamma)` | `MHD_NC` | `rho`, `u`, `v`, `w`, `P`, `Bx`, `By`, `Bz`, `B2_over_2` |
| `dump_imhd_glm_frame(path, mesh, q, gamma)` | `MHD_GLM_NC` | the MHD arrays plus `psi` |
| `dump_maxwell_frame(path, mesh, q)` | `MAXWELL_NC` | `E_x`, `E_y`, `E_z`, `B_x`, `B_y`, `B_z`, `E_energy`, `B_energy` |

`P` is the gas pressure (for MHD, total energy minus kinetic and
magnetic energy, times $\gamma - 1$). `B2_over_2` is the magnetic
pressure $|\mathbf{B}|^2/2$. `E_energy` and `B_energy` are
$|\mathbf{E}|^2/2$ and $|\mathbf{B}|^2/2$. No floors are applied: the
presets divide by the stored density.

For any other field set, call `write_vtu` with your own closure.

## File format

* XML `UnstructuredGrid` with every array in one raw appended-binary
  section: `format="appended"`, `byte_order="LittleEndian"`,
  `header_type="UInt64"`. Each block is an 8-byte length followed by the
  values. Points and point data are `Float64`, connectivity and offsets
  `Int32`, cell types `UInt8`.
* Each $(P+1)^d$-node element is split into $P^d$ linear sub-cells, so
  the file shows the DG polynomial as a piecewise-linear field at any
  order:

  | Dim | Sub-cell | VTK cell type |
  | --- | -------- | ------------- |
  | 2   | quad     | `VTK_QUAD = 9` |
  | 3   | hex      | `VTK_HEXAHEDRON = 12` |

* Every element owns its own copy of its points, so jumps across element
  faces are kept rather than averaged. 2D points are written with
  $z = 0$.

`tests/test_vtu_writers.mojo` reads the files back and checks the
layout block by block and every array against an analytic field.

## `FrameWriter`

```mojo
from warpxm.io.frame_writer import FrameWriter

var fw = FrameWriter("sod_2d")          # frames go to ./output/sod_2d/

var t: Real = 0.0
for frame in range(NUM_FRAMES):
    dump_euler_frame(fw.next_path(t), mesh, q, gamma)
    if frame < NUM_FRAMES - 1:
        t = stepper.run_until(
            ctx, q, t, Real(frame + 1) * frame_dt, dt, compute_rhs, adjust,
        )
fw.write_pvd()
```

* `FrameWriter(basename, directory="")`: with no directory, frames go to
  `output/<basename>` relative to the working directory. The directory
  is created if it does not exist. `/output/` is in `.gitignore`.
* `fw.next_path(t)` returns `<directory>/<basename>_0000.vtu` (four-digit
  frame number) and records the time `t`.
* `fw.write_pvd()` writes `<directory>/<basename>.pvd`, a ParaView
  collection listing every frame with its time. Open that file for a
  time slider.
* `fw.directory`, `fw.frame_files` and `fw.frame_times` are readable.

## ParaView usage

Open the `.pvd`, hit Apply, colour by field. See
[Viewing output in ParaView](../getting-started/paraview.md).
