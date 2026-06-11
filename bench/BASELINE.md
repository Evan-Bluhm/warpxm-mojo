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
