# Installation

warpxm-mojo is GPU-only. `has_accelerator()` is a **compile-time
precondition** in every entry point — there is no CPU fallback path.

## Supported hardware

| Vendor | Status |
| ------ | ------ |
| NVIDIA (CUDA-capable, compute capability ≥ 7.0, e.g. V100, A100, H100, RTX) | ✅ primary target |
| AMD CDNA (MI200/MI300 series) | ✅ via Mojo's AMD backend |
| Apple silicon (M-series GPU) | ✅ via Mojo's Metal backend |

## Prerequisites

* **pixi** — single binary, [install instructions on pixi.sh](https://pixi.sh/latest/#installation).
  pixi manages the conda channels (`max-nightly`, `conda-forge`) that
  ship Mojo and resolves a lock file so every checkout gets the same
  toolchain.
* A working GPU driver. Confirm with `nvidia-smi` (NVIDIA),
  `rocm-smi` (AMD), or — on macOS — `system_profiler SPDisplaysDataType`.

That's it. You don't need to install Mojo, MAX, CUDA, or anything else
separately.

## First-time setup

```fish
git clone https://github.com/<your-fork>/warpxm-mojo.git
cd warpxm-mojo
pixi install
pixi run mojo --version
```

`pixi install` resolves the toolchain into `.pixi/envs/default/`. The
first run is slow because it downloads Mojo; subsequent runs are
instantaneous.

`pixi run mojo --version` prints the version of the bundled Mojo
compiler — useful to include in bug reports.

## Confirm the GPU is visible

```fish
pixi run mojo run -I src tests/test_advection_1d.mojo
```

This is the smallest end-to-end test. If you see `PASSED`, you have a
working build with a visible GPU. If it fails with `Requires GPU` or a
similar message, your `pixi` environment is fine but Mojo can't find a
device — check vendor-specific drivers (`nvidia-smi`, `rocm-smi`).

## The `pixi run` incantation

Every command runs through pixi so it has the right `PATH`, library
paths, and Mojo version. The convention used everywhere in this repo:

```fish
pixi run mojo run -I src <file>.mojo
```

`-I src` adds `src/warpxm/` to Mojo's module search path so `from
warpxm.… import …` lines resolve. **Don't drop `-I src`** — you'll get
mysterious "module not found" errors.

## Optional: documentation toolchain

If you want to preview this site locally:

```fish
pixi run --environment docs docs-serve
```

This spins up Zensical's dev server at <http://localhost:8000>. The
`docs` feature is gated separately so the default GPU environment stays
lean.

## Next

[Run your first example →](first-example.md)
