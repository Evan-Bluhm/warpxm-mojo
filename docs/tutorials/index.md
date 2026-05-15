# Tutorials

End-to-end walkthroughs of polished example drivers, each showing how a
specific piece of the framework fits together. Pick whichever maps
closest to what you want to do.

| Tutorial | Physics | Features showcased |
| -------- | ------- | ------------------ |
| [Sod tube](sod-tube.md) | Euler 2D | Shocks, Moe-Rossmanith limiter, VTU output |
| [Orszag-Tang vortex](orszag-tang.md) | GLM-MHD 2D | Divergence cleaning, periodic time-series, integrated diagnostics |
| [TM cavity mode](cavity-mode.md) | Maxwell 2D / 3D | PEC walls, energy conservation, ∫E_energy ↔ ∫B_energy oscillation |
| [Sedov blast with positivity](sedov-blast.md) | Euler 2D | Strong shock, Zhang-Shu positivity preserver, conservation diagnostics |

If you haven't run anything yet, start with [Getting Started → First example](../getting-started/first-example.md)
which walks through the simpler `examples/advection_1d.mojo`.
