# Viewing output in ParaView

Every 2D and 3D physics module has a VTU preset on top of one generic
writer. Examples whose names
end in `_vtu.mojo` dump a sequence of `.vtu` frames plus a `.pvd`
collection file that ParaView reads as a single time-varying dataset.

## Generating frames

Pick a VTU-equipped example and run it:

```fish
pixi run example imhd_orszag_tang_2d_vtu
```

The frames go into `./output/<name>/` under the directory you ran from
(here `output/orszag_tang/`; the example prints the path). The directory
is created for you. After the run it contains:

```text
orszag_tang_0000.vtu
orszag_tang_0001.vtu
...
orszag_tang_0010.vtu
orszag_tang.pvd        ← time-series collection
```

The files are binary (raw appended data, little-endian `Float64`), so
they are compact and load quickly but are not meant to be read in a text
editor.

## Loading in ParaView

1. **File → Open** and choose the `.pvd` file (not an individual
   `.vtu`). Loading the `.pvd` gives you the time slider.
2. Click **Apply** in the Properties panel. ParaView reads frame 0.
3. In the **Coloring** dropdown, pick the field — `rho`, `Bx`, `psi`,
   `E_energy`, etc., depending on the example.
4. Hit **Play** ▶ to step through frames.

## Per-physics field menus

Each preset writes the same arrays in 2D and 3D:

| Preset | Fields |
| ------ | ------ |
| `dump_euler_frame` | `rho`, `u`, `v`, `w`, `P` |
| `dump_imhd_frame` | `rho`, `u`, `v`, `w`, `P`, `Bx`, `By`, `Bz`, `B2_over_2` |
| `dump_imhd_glm_frame` | the MHD fields plus `psi` |
| `dump_maxwell_frame` | `E_x`, `E_y`, `E_z`, `B_x`, `B_y`, `B_z`, `E_energy`, `B_energy` |

To write other quantities, call `write_vtu` with your own field closure;
see [`io`](../api/io.md).

## Working with the DG mesh

Each (P+1)ᵈ-node DG element is decomposed into P^d linear sub-cells
(quads in 2D, hexes in 3D) so the writer is order-agnostic. Element
boundaries show the DG discontinuity directly — no averaging is
applied. For smooth fields the jumps look like aliasing artefacts; for
shocks they're the correct physical picture.

If you want a continuous render you can apply ParaView's **Cell Data to
Point Data** filter, but be aware it averages across the discontinuities.

## Tip: extracting a 1D line

For 1D-extruded problems (Brio-Wu, Lax) the easiest cross-section is
`Plot Over Line` from the Filters menu, with the line endpoints at
(0, y_mid, …) → (1, y_mid, …). That gives you the classic x-vs-ρ
plot directly.
