# Design departures from upstream

warpxm-mojo is a *port* of [WARPXM](https://github.com/csumlak/warpxm),
not a translation. Two design decisions diverge from upstream
deliberately. Everything else — the physics modules, the DG
formulation, the SSPRK3 and limiter algorithms — is faithful.

## 1. GPU only

There is no CPU spatial solver, no Kokkos / non-Kokkos build split, no
runtime backend dispatch. Every piece of physics has a single device
implementation.

```mojo
def main() raises:
    comptime assert has_accelerator(), "warpxm-mojo requires a GPU"
    ...
```

`has_accelerator()` is a Mojo intrinsic that returns true at
compile-time when targeting a GPU (NVIDIA, AMD, Apple silicon). The
`comptime assert` makes the lack of a GPU a build error, not a runtime
error. This is intentional: the host-side reference implementation in
upstream WARPXM exists for development convenience but is the second
codebase that has to track every algorithmic change. Removing it
means:

* one implementation to maintain per kernel,
* no risk of CPU / GPU drift,
* aggressive use of Mojo's `comptime if is_nvidia_gpu()` for backend
  specialisations where they matter (rare so far — most kernels are
  portable across vendors).

If you need a CPU fallback for development on a laptop without a GPU,
the recommended path is to run the same Mojo programs on an Apple
silicon Mac — every M-series chip has a Metal-capable GPU that Mojo
targets natively.

## 2. Compiled input files

Upstream WARPXM ships an executable that loads Python-generated input
descriptions at runtime. A typical Python input file declares a mesh, a
list of variables, a list of solvers, hooks them together, and pickles
the result for the C++ runtime to read.

warpxm-mojo replaces that entirely. An "input file" is now a Mojo
program that imports the solver modules and calls into them:

```mojo
from warpxm.apps.euler import Euler, EULER_NC
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
...
def main() raises:
    var mesh = BlockMesh2D[P](ctx, ...)
    var app = Euler(gas_gamma=1.4, ...)
    var spatial = DGEulerSolver2D[P](app)
    ...
```

The executable for a given physics problem is built by compiling its
driver file. Consequences:

* **No `WxCryptSet`**, no JSON/INI/HDF5 input deserialiser, no
  type-erased variable registry. A bad parameter is a compile error.
* **Compile-time specialisation everywhere.** Polynomial order `P`,
  component count `NC`, mesh dimension — all `comptime` parameters
  baked into kernels. No runtime branches on these in the hot path.
* **No batching different problems in one binary.** That's by design;
  each problem you'd run is small enough to keep its driver self-
  contained, and you save the entire input-file plumbing.

The trade-off: to change a parameter you re-compile. In practice that's
a few seconds on the second build (Mojo caches aggressively).

## What stays the same

* **Algorithms.** DG strong form, SSPRK3, Moe-Rossmanith limiter,
  Dedner GLM cleaning, Rusanov / HLLD numerical fluxes, source-
  coupled stepper for plasma. All match upstream.
* **Module organisation.** `apps/`, `spatial/`, `temporal/`,
  `variable/`, `adjusters/`, `driver/` mirror the upstream subtrees of
  the same names (under `src/apps`, `src/dfem/spatial_solvers/`,
  `src/dfem/temporal_solvers/`, `src/dfem/variable_adjusters/`).
* **Test problems.** Sod tube, Brio-Wu, Orszag-Tang, cavity TM modes,
  cyclotron orbits — all the canonical validations from upstream are
  here.

If you know upstream WARPXM, finding the equivalent file in
warpxm-mojo is almost always a one-step lookup: the file name with
`_kokkos.cc` stripped and `.cc` → `.mojo`.
