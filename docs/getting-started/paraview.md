# Viewing output in ParaView

Every 2D and 3D physics module has a VTU writer. Examples whose names
end in `_vtu.mojo` dump a sequence of `.vtu` frames plus a `.pvd`
collection file that ParaView reads as a single time-varying dataset.

## Generating frames

Pick a VTU-equipped example, create the output directory it mentions in
its docstring, then run:

```fish
mkdir -p /tmp/orszag_tang
pixi run mojo run -I src examples/imhd_orszag_tang_2d_vtu.mojo
```

By default the output goes into `/tmp/<problem-name>/`. After the run
that directory contains:

```text
orszag_tang.0000.vtu
orszag_tang.0001.vtu
...
orszag_tang.0010.vtu
orszag_tang.pvd        ← time-series collection
```

## Loading in ParaView

1. **File → Open** and choose the `.pvd` file (not an individual
   `.vtu`). Loading the `.pvd` gives you the time slider.
2. Click **Apply** in the Properties panel. ParaView reads frame 0.
3. In the **Coloring** dropdown, pick the field — `rho`, `Bx`, `psi`,
   `E_energy`, etc., depending on the example.
4. Hit **Play** ▶ to step through frames.

## Per-physics field menus

The VTU writers expose fields tuned to each module:

| Writer | Fields |
| ------ | ------ |
| `vtu_2d` / `vtu_3d` (Euler) | `rho`, `u`, `v`, `w` (3D), `P` |
| `vtu_imhd_2d` / `vtu_imhd_3d` (ideal MHD / GLM) | `rho`, `u`, `v`, `w` (3D), `P`, `Bx`, `By`, `Bz` (3D), `B2_over_2`, `psi` (GLM) |
| `vtu_maxwell_2d` / `vtu_maxwell_3d` | `E_x`, `E_y`, `E_z`, `B_x`, `B_y`, `B_z`, `E_energy`, `B_energy` |

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
