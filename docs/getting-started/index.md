# Getting started

You'll need a Linux or macOS machine with an Apple-silicon or NVIDIA GPU
(see [GPU architecture in Mojo](../concepts/gpu-architecture.md) for the
exact list of supported targets). Everything else — Mojo itself, numpy,
the test runner — is materialised by **pixi**.

The shortest path:

```fish
git clone https://github.com/<your-fork>/warpxm-mojo.git
cd warpxm-mojo
pixi install
pixi run mojo run -I src examples/advection_1d.mojo
```

If the last command prints a periodic-advection convergence run, you're
fully set up.

## Three pages from here

1. [Installation](installation.md) — pixi, Mojo, what GPUs are supported,
   how to confirm the GPU is visible.
2. [Running your first example](first-example.md) — a guided tour of
   `examples/advection_1d.mojo`: what it computes, how the mesh / state /
   stepper fit together, and what the output means.
3. [Viewing output in ParaView](paraview.md) — every 2D / 3D physics has
   a VTU writer. This page shows how to load a `.pvd` time series and
   colour by ρ or |B|.
