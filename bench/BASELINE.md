# DG-core perf baseline

Canonical metric: **`steps/s` at fixed mesh size**. Higher is better.

Acceptance for any DG-core change: no regression > 5 % on any row of
this table (run `scripts/bench.sh` and diff `bench/results.csv` against
the numbers below). A win on one config that costs > 5 % on another is
also a regression — Tasks 1 + 2 in `tasks/dg-core-performance.md` are
expected to lift *every* row.

## Run details

| Field           | Value                          |
| --------------- | ------------------------------ |
| Date            | 2026-05-17                     |
| Commit          | `d58e914` (pre-task-1)         |
| GPU             | NVIDIA GeForce RTX 3090        |
| Driver          | 596.36                         |
| Mojo            | pinned via `pixi.lock`         |
| Window          | 10 warmup + 100 measured steps |
| Reproduce       | `scripts/bench.sh`             |

Mesh sizes are fixed across the 4 physics for fair comparison within
each dimensionality:

- 2D: `NX=NY=128, P=2` (16 384 elements)
- 3D: `NX=NY=NZ=32, P=2` (32 768 elements)

## Numbers — baseline (commit `d58e914`)

| Bench         | NC | steps/s  | elapsed (100 steps) |
| ------------- | -: | -------: | ------------------: |
| advection_2d  | 1  | 1369.51  | 0.073 s             |
| advection_3d  | 1  | 461.46   | 0.217 s             |
| euler_2d      | 5  |  94.32   | 1.060 s             |
| euler_3d      | 5  |  30.18   | 3.314 s             |
| imhd_2d       | 8  |  73.74   | 1.356 s             |
| imhd_3d       | 8  |  22.90   | 4.367 s             |
| imhd_glm_2d   | 9  |  72.91   | 1.372 s             |
| imhd_glm_3d   | 9  |  24.59   | 4.067 s             |

Total bench wall-clock: ~16 s on this hardware.

## Numbers — after Task 1 (element packing)

`ELEMS_PER_BLOCK = 4` on the 2D core, `= 4` on the 3D core (block_dim.z
expanded to `(P+1) · EB`), `= 4` on the 1D core (a second thread axis
gives `(P+1) · EB = 12` threads/block). EB=8 and 16 in 1D / 3D triggered
codegen hangs on specific Vlasov-Maxwell coupled tests — EB=4 is the
conservative ceiling and Mojo-codegen-flaky, not fundamental.

| Bench         | NC | steps/s  | speedup vs baseline |
| ------------- | -: | -------: | ------------------: |
| advection_2d  | 1  | 2743.66  | **2.00×**           |
| advection_3d  | 1  |  466.29  |  1.01×              |
| euler_2d      | 5  |  196.18  | **2.08×**           |
| euler_3d      | 5  |   32.15  |  1.07×              |
| imhd_2d       | 8  |  159.36  | **2.16×**           |
| imhd_3d       | 8  |   24.27  |  1.06×              |
| imhd_glm_2d   | 9  |  152.09  | **2.09×**           |
| imhd_glm_3d   | 9  |   25.88  |  1.05×              |

## Numbers — after Task 2 (single-accumulator on 3D core only)

Task 2's single-accumulator pattern folds `dfdx`, `dgdy`, `dhdz` (and
`surface` in 3D) into one running `accum` array — dropping the live
set by ~2·NC doubles per thread. Applied only to `dg_core_3d.mojo`.
On `dg_core_2d.mojo` the same change broke `test_euler_123_2d`
(positivity preserver couldn't recover from the FP-reordered
near-vacuum trajectory) and `test_imhd_briowu_2d_hlld` (y-drift
bit-identity); the 2D speedup ceiling from Task 1 alone is already
2.0–2.2×, so Task 2 on 2D was reverted. 3D had no analogous test
breakage.

| Bench         | NC | steps/s  | speedup vs baseline | vs Task 1 only |
| ------------- | -: | -------: | ------------------: | -------------: |
| advection_2d  | 1  | 2767.92  | **2.02×**           | (unchanged)    |
| advection_3d  | 1  |  602.98  | **1.31×**           | **1.31×**      |
| euler_2d      | 5  |  201.60  | **2.14×**           | (unchanged)    |
| euler_3d      | 5  |   44.85  | **1.49×**           | **1.39×**      |
| imhd_2d       | 8  |  159.37  | **2.16×**           | (unchanged)    |
| imhd_3d       | 8  |   32.87  | **1.44×**           | **1.35×**      |
| imhd_glm_2d   | 9  |  154.03  | **2.11×**           | (unchanged)    |
| imhd_glm_3d   | 9  |   31.61  | **1.29×**           | **1.22×**      |

**3D goal (≥ 1.3× over baseline): met for euler_3d / imhd_3d
(1.44–1.49×), just under for imhd_glm_3d (1.29×).** The 9-component
GLM state is the worst-case for register pressure; squeezing the
last bit will need Task 2's other levers (`q_neg` / `q_pos` /
`f_star` to shared, `@noinline` on HLLD inner) — out of scope for
this pass.

**`advection_3d` (1.31×)**: refactored `spatial/dg_3d.mojo` from its
inline standalone kernel into a thin wrapper over
`_dg_3d_compute_rhs_core` (mirroring the 2D advection wrapper). Now
benefits from the same Task 1 + Task 2 wins as the other 3D physics
— the speedup matches what the shared core gives for NC=1.

## Reading the table

A few observations against the [DG-core-performance task plan]
(../tasks/dg-core-performance.md):

- **NC scaling within a dimension is steep.** At baseline, 2D goes
  `1369 → 94 → 74 → 73` steps/s as NC climbs `1 → 5 → 8 → 9`. After
  Task 1: `2744 → 196 → 159 → 152` — the curve has the same shape but
  shifted up by a constant ~2× factor, confirming Task 1 wins are
  uniform across NC (register pressure didn't budge; warp utilisation
  did).

- **2D-vs-3D gap stayed ~6×** after Task 1 (advection: 2744/466,
  imhd: 159/24). At baseline it was ~3×. Element packing helped 2D
  more than 3D — exactly as the register-pressure analysis predicted.

- **GLM-MHD ≈ MHD steps/s** in both rows after Task 1
  (`imhd_glm_2d = 152` vs `imhd_2d = 159`). The GLM source kernel +
  ψ component (NC=9 vs 8) costs ~5 % vs straight MHD. Useful sanity
  check that GLM isn't a regression vector.

## Numbers — after the FP64 trim (2026-11-21, Mojo 1.2.0)

Prompted by upstream WARPXM's kernel work (`143dcfde0`, `a8cce53c7`,
`tasks/KERNEL_PERF.md` there), which found its kernels bound by per-thread
local-memory traffic. Profiling ours with `ncu` showed a different
picture: local memory was already under 1 % of instructions, and every DG
right-hand-side kernel had the FP64 pipe 87–88 % busy. So the lever here
is FP64 instruction count, and three things dominated it:

1. `math_compat.sqrt` was `x ** 0.5`, a software `pow` — two per Euler
   face evaluation, four per MHD one. Now the `llvm.sqrt` intrinsic.
2. Divisions: the 2D core divided by the Jacobian once per component
   (~73 double divisions per thread for MHD), and the fluxes divided by ρ
   separately for each velocity component. Now one reciprocal per state,
   `1 / jacobian` from the launcher, comptime reciprocal weights.
3. The Rusanov flux built f_x, f_y, f_z for both sides and dotted them
   with the normal; the two off-axis fluxes were multiplied by a literal
   zero that IEEE arithmetic cannot fold. Now F·n is formed directly.

Also removed the remaining local-memory traffic (`diff_entry` select
instead of the non-inlined `diff_matrix_flat()` call); that made no
measurable difference on this GPU.

Same session, same GPU, best of two runs, before = commit `f4531fa`:

| Bench         | NC | before steps/s | after steps/s | speedup |
| ------------- | -: | -------------: | ------------: | ------: |
| advection_2d  | 1  | 2596.8         | 6241.1        | 2.40×   |
| advection_3d  | 1  |  580.1         |  974.3        | 1.68×   |
| euler_2d      | 5  |  181.5         |  683.9        | 3.77×   |
| euler_3d      | 5  |   38.5         |  109.7        | 2.85×   |
| imhd_2d       | 8  |  121.6         |  364.8        | 3.00×   |
| imhd_3d       | 8  |   23.5         |   59.1        | 2.51×   |
| imhd_glm_2d   | 9  |  119.0         |  334.0        | 2.81×   |
| imhd_glm_3d   | 9  |   23.0         |   53.0        | 2.30×   |

Per-launch `ncu` counters for the RHS kernel (thread instructions):

| Kernel   | time before → after | FP64 instr.      | double divisions |
| -------- | ------------------- | ---------------- | ---------------- |
| euler_2d | 1.62 → 0.40 ms      | 111.7M → 28.3M   | 7.36M → 0.34M    |
| imhd_2d  | 3.08 → 0.76 ms      | 193.7M → 53.2M   | 10.78M → 0.34M   |
| euler_3d | 8.61 → 2.44 ms      | 734.3M → 235.0M  | 39.81M → 2.65M   |
| imhd_3d  | 14.41 → 4.70 ms     | 1126.9M → 439.1M | 38.47M → 2.65M   |

The FP64 pipe is still ~87 % busy afterwards, so further gains on this
class of GPU have to come from fewer FP64 instructions again (the face
flux is evaluated once from each side of every face), not from memory
layout. On GPUs with full-rate FP64 the balance will differ; re-profile
there before drawing conclusions.

The MHD positivity preserver was a separate bottleneck in limited runs:
2.46 ms per call against 0.36 ms for the RHS in the 3D Orszag-Tang test,
because it bisected 50 times for every node of any element that needed
limiting. It now bisects only nodes that still violate the bound at the
current θ: 0.19 ms per call, same result to round-off.

## Vlasov-Maxwell 1D2V (2026-11-21)

Driver: `profiling/prof_vlasov_1d2v.mojo` — 64 × 32 × 32 phase-space
elements at P = 2 (1.77M nodes), coupled SSPRK3 step. Per-launch numbers
from `ncu` on the RTX 3090:

| Kernel                    | before   | after    | what changed |
| ------------------------- | -------: | -------: | ------------ |
| kinetic RHS               | 1.549 ms | 0.555 ms | no local memory, no double divisions |
| first moment (v1)         | 0.867 ms | 0.064 ms | 32 lanes per x-node + shared-memory combine |
| first moment (v2)         | 0.710 ms | 0.055 ms | same |
| whole step                | 111 steps/s | 444 steps/s | **4.0×** |

This is the kernel where upstream's diagnosis applied directly. The
kinetic RHS wrote 464 MB to DRAM per launch for a 14 MB result: the three
differentiation matrices and three weight tables were returned by
non-inlined calls into per-thread local memory (92 registers, 1.77M local
stores per launch). With `diff_entry`, comptime reciprocal weights and
host-side `1 / jacobian` and `(q/m) / λ_skin` it writes 12 MB, uses 53
registers and has no local loads or stores.

The moment kernels were the other half of the step: one thread per x-node
looping serially over all 9216 velocity nodes, so 192 threads on the whole
GPU. They are now one generic kernel (`_moment_kernel_1d2v`) with 32 lanes
per x-node.

What is left in the kinetic RHS is about 360 instructions per thread, of
which only ~40 are FP64; the rest is 64-bit index arithmetic and selects.
Launching on a (num_v2, num_v1, num_x) grid to avoid unpacking the block
index removed instructions but did not change the time. The 1D1V RHS got
the same treatment; the 1D1V moment kernels are still one thread per
x-node (their velocity loop is two orders of magnitude shorter).

## Dimension-generic core (2026-11-22)

`dg_core_{1,2,3}d.mojo` were merged into one `dg_core.mojo`, generic over
`NDIM` with `comptime for` over the axes. The 3D single-accumulator and the
1D / 2D per-axis derivative arrays are kept as they were (`SINGLE_ACCUMULATOR`
in the core), and the right-hand side is bit-identical to the old cores in
all three dimensions. Interleaved A/B on the same machine state, old vs new
(steps/s, three runs each):

| Bench         | old                 | new                 |
| ------------- | ------------------- | ------------------- |
| advection_3d  | 915 / 939 / 835     | 902 / 899 / 917     |
| euler_2d      | 668 / 658 / 650     | 668 / 664 / 665     |
| imhd_2d       | 354 / 341 / 361     | 356 / 359 / 347     |
| imhd_3d       | 56.9 / 55.1 / 57.3  | 55.7 / 57.3 / 56.6  |

The merge also found a deadlock in the element packing from Task 1: the
guard for a partly filled last block was `barrier(); return`, which puts
the barrier on a divergent path and hangs the launch whenever the element
count is not a multiple of `ELEMS_PER_BLOCK`. Every mesh in the suite
happened to have a multiple of 4 elements. This is very likely what the
"codegen hangs" at EB = 8 and 16 noted above were (meshes whose element
count was a multiple of 4 but not of 8), so larger EB values are worth
re-measuring now that the guard is fixed.
