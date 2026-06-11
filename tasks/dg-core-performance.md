# DG core performance — upcoming work

Profiling on 2026-05-16 (RTX 3090, nsys 2025.5.2 + ncu 2025.4.0) showed
the per-physics DG kernels through `_dg_<N>d_compute_rhs_core` consume
**90–95 % of GPU time** in every problem we run, and that they hit a
**16.67 % theoretical-occupancy ceiling** regardless of mesh size,
because:

- Each block runs one element, so `block_dim = (P+1)^NDIM` —
  9 threads (2D, P=2), 27 threads (3D, P=2). Neither aligns with the
  32-lane warp.
- The compiler spills to the max **255 registers / thread**, capping
  the SM at 8 resident blocks (`Block Limit Registers = 8`).
- Result: SM compute throughput 87 %, DRAM 1.3–4.5 %, L1/TEX 1.6 %.
  Pipes are busy issuing, but warps stall ~54 of 65 cycles on the
  shared-memory MIO scoreboard and there aren't enough warps to hide
  it.

Everything else (limiter 3–8 %, sources <1 %, steppers ~1 %) is noise
by comparison and should not be optimised until the DG core is fixed.
Raw artifacts live under `profiling/` — `imhd_glm_2d.nsys-rep`,
`imhd_3d.nsys-rep`, `ncu_dg_imhd_glm_2d.txt`, `ncu_dg_imhd_3d.txt`,
plus the two stepping-only drivers used to collect them.

The upcoming work is broken into five tasks below, ordered by leverage
and dependency. Tasks 1 + 2 are the structural fixes; Task 0 puts the
measurement harness on a permanent footing so 1 and 2 can be evaluated
honestly; Tasks 3 and 4 are follow-ups.

---

## Task 0 — Lock in a perf-regression harness  ✅ LANDED 2026-05-17

The harness shipped exactly as described below. Eight bench drivers
under `bench/`, an aggregator at `scripts/bench.sh`, and a baseline
at `bench/BASELINE.md`. `scripts/bench.sh` runs end-to-end in ~16 s
on an RTX 3090 and writes `bench/results.csv` for diffable
regression checks.

Headline numbers (RTX 3090, commit `d58e914`):
- 2D `(NX=NY=128, P=2)`: advection 1369 / euler 94 / imhd 74 / glm 73 steps/s
- 3D `(NX=NY=NZ=32, P=2)`: advection 461 / euler 30 / imhd 23 / glm 25 steps/s

Read the rest of this section for the historical brief; the file
exists for Task 1 / Task 2 to consume.



**Why first.** Tasks 1 and 2 will rewrite the hottest kernel in the
codebase. Without a reproducible perf number per configuration we
can't tell a 1.4× win from a 1.05× win from noise, and we can't tell
whether a "win" only helps one (NDIM, P, NC) combination at the
expense of another.

**What to build.**

1. Promote the existing stepping-only drivers
   (`profiling/prof_imhd_glm_2d.mojo`, `profiling/prof_imhd_3d.mojo`)
   from ad-hoc scripts into a small fixed set of benchmark drivers
   under `bench/`, one per representative (physics, NDIM, P, NC):
   - `bench_advection_2d.mojo`, `bench_advection_3d.mojo` (NC = 1,
     cheap closures — isolates DG-core scaling from physics).
   - `bench_euler_2d.mojo`, `bench_euler_3d.mojo` (NC = 5).
   - `bench_imhd_2d.mojo`, `bench_imhd_3d.mojo` (NC = 8).
   - `bench_imhd_glm_2d.mojo`, `bench_imhd_glm_3d.mojo` (NC = 9).
   Each runs a fixed warmup (10 steps) + measure window (100 steps),
   prints elapsed wall-clock and `steps/s` to stdout, and does no host
   I/O between the two `ctx.synchronize()` calls.
2. Add `scripts/bench.sh` that runs all eight, captures stdout into a
   single CSV with columns `(config, steps, elapsed_s, steps_per_s)`,
   and optionally `--profile` to also emit nsys traces under
   `profiling/`.
3. Document the canonical metric: **`steps/s` at fixed mesh size**.
   Record the current baseline in `bench/BASELINE.md` alongside the
   commit SHA, GPU model, and date. Future commits update the table.

**Acceptance.**
- `scripts/bench.sh` runs end-to-end in under a few minutes.
- `bench/BASELINE.md` has a row per benchmark with the current number.
- A regression of >5 % on any benchmark is detectable from a diff of
  the CSV.

**Risk / non-goals.** This task adds zero performance. Resist the
temptation to fold in optimization here — keep it strictly about
measurement.

---

## Task 1 — Pack multiple elements per block in the DG core  ⚠️ PARTIAL — LANDED 2026-05-17

Element packing landed across all three cores
(`dg_core_{1,2,3}d.mojo`) and the 13 per-physics wrappers that use
them (the 14th — `dg_3d.mojo` advection — has a standalone kernel and
needs a separate pass).

| Cell                | Target  | Achieved      |
| ------------------- | ------: | ------------: |
| 2D ≥ 1.5× steps/s   |    1.5× | **2.00–2.16×** ✓ |
| 3D ≥ 1.3× steps/s   |    1.3× | 1.05–1.07× ✗  |
| advection_3d        |    n/a  | 1.01× (standalone kernel — not touched yet) |

**Why the 3D target was missed.** The 3D kernel is already register-
bound at 255 reg/thread before any packing. Element packing improves
warp utilisation (more work per launch) but doesn't move the
occupancy needle, which is set by registers and shared memory.
Task 2 (register cut) is the lever for 3D.

**Settings shipped:**
- 2D: `DG_CORE_2D_ELEMS_PER_BLOCK = 4` (block_dim = (P+1, P+1, 4))
- 3D: `DG_CORE_3D_ELEMS_PER_BLOCK = 4` (block_dim = (P+1, P+1, (P+1)·4))
- 1D: `DG_CORE_1D_ELEMS_PER_BLOCK = 4` (block_dim = (P+1, 4))

**Codegen flakiness to remember.** EB=8 / 16 in 1D triggered hangs on
Vlasov-Maxwell coupled tests (`test_vlasov_maxwell_uniform_oscillator_
1d1v`); EB=4 is the conservative ceiling. The Mojo `if cond:
reassign_InlineArray` pattern after `barrier()` *also* breaks
silently — `if cond: barrier(); return` is the safe idiom for
partial-last-block guards. Both behaviours documented in the core
files.

**Acceptance status.**
- All 87 / 87 regression tests pass bit-for-bit (or within their
  tolerances — bit-identity tests like
  `test_imhd_briowu_2d_hlld`'s y-drift assertions still hold).
- `bench/BASELINE.md` updated with the after-Task-1 numbers.
- Original Task-1 brief follows for the Task 2 author.



**Goal.** Eliminate the warp-size mismatch by tiling N elements into
one CUDA block so the block size becomes a useful multiple of 32.
Expected to land most of the available speedup on its own.

**Files.** `src/warpxm/spatial/dg_core_1d.mojo`,
`src/warpxm/spatial/dg_core_2d.mojo`,
`src/warpxm/spatial/dg_core_3d.mojo`, and the per-physics wrapper
`enqueue_function` calls in `src/warpxm/spatial/dg_<phys>_<N>d.mojo`
(all five physics × three dimensions = ~15 wrapper sites).

**Approach.**

1. Parameterise the core on a new `comptime ELEMS_PER_BLOCK: Int`
   (default 1 → byte-identical to today's behaviour, useful as a
   compile-time off switch during bring-up).
2. Change the block layout so `block_idx.x = elem_block`,
   `thread_idx.{x,y,z}` covers `(local_node_dims..., elem_in_block)`:
   - 2D P=2: `block_dim = (P1, P1, ELEMS_PER_BLOCK)` —
     4 elements per block ⇒ 36 threads ≈ 1.13 warps. 8 elements ⇒
     72 threads ≈ 2.25 warps. The sweet spot is whichever value
     keeps register pressure low enough to still hit ≥ 32 resident
     warps per SM.
   - 3D P=2: `block_dim = (P1, P1, P1*ELEMS_PER_BLOCK)` or split
     across `block_dim.x/y/z` — pick whichever Mojo's launch
     constraints allow.
   - 1D P=2: `block_dim = (P1, ELEMS_PER_BLOCK)` — 1D has the worst
     starting point (3 threads/block today), so the relative win
     here is largest.
3. Shared-memory layout in `_dg_<N>d_compute_rhs_core` must now be
   indexed by `(elem_in_block, local_node, c, flux_axis)`. Bump the
   `stack_allocation[..., row_major[...]]` size accordingly.
4. The `face_neg_elem` / `face_pos_elem` / `face_bc` lookups today
   index by `e * 4 + FACE_DIR_*` (2D); these still work — each thread
   computes its own `e = elem_block * ELEMS_PER_BLOCK + elem_in_block`
   and the surface branches stay unchanged.
5. Grid size in the per-physics wrappers becomes
   `ceildiv(mesh.num_elements, ELEMS_PER_BLOCK)`. Don't forget the
   boundary check at the top of the kernel — `if e >= num_elements:
   return` still applies and is now load-bearing for the final
   partial block.

**Acceptance.**
- All 70 regression tests in `tests/` pass bit-for-bit (or within the
  tolerances the tests already check — many test exact symmetries).
- `scripts/bench.sh` shows ≥ 1.5× `steps/s` on the four 2D benchmarks
  and ≥ 1.3× on the four 3D benchmarks. (Conservative — the back-of-
  envelope upper bound from going 9→32 lanes is ~3.5×, but register
  pressure is unchanged so achieved occupancy won't budge yet.)
- ncu reports `Block Size` as a multiple of 32 (or at least ≥ 32) and
  `Achieved Active Warps Per SM` rises.
- DRAM throughput stays low — if it spikes above ~20 %, packing is
  introducing redundant loads; revisit the shared-memory layout.

**Risk / things to watch.**
- **Increased shared-memory per block** could become the binding
  occupancy constraint instead of registers. Today's static shmem is
  1296 B (2D) / 5184 B (3D). At `ELEMS_PER_BLOCK = 4` that goes to
  ~5 KB / ~20 KB — still well under the 100 KB/SM limit, but if
  someone bumps `ELEMS_PER_BLOCK = 16` it can bite. Check
  `Block Limit Shared Mem` in ncu after the change.
- **Within-warp divergence at element edges.** The surface branches
  (`if in_ == 0`, `if in_ == P`, etc.) were already divergent inside a
  block; now they also vary across the `elem_in_block` axis, but they
  still vary deterministically — the warp executes both sides
  serially as it does today. No new pathology, but confirm with the
  ncu "Branch Efficiency" metric.
- **Edge case for `num_elements` not divisible by `ELEMS_PER_BLOCK`.**
  The partial last block must still write correct values for in-range
  elements without writing past `rhs`. Covered by the existing
  boundary check.

**Validation plan.**
1. Start with 2D advection (NC=1, simplest closures). Compare bit-for-
   bit against the baseline via `tests/test_advection_2d.mojo` and the
   3 convergence orders in `test_dg_convergence_2d.mojo`. If those
   pass, move on.
2. Then 2D Euler / IMHD / IMHD-GLM, each gated by the matching test
   subset. Bit-identity for periodic smooth ICs (Alfvén, entropy
   wave); within-tolerance for shock problems (Brio-Wu, Orszag-Tang,
   blast).
3. Repeat for 3D.

**Why bit-identity matters.** Several tests explicitly verify
extruded-1D-in-2D produces a y-drift of exactly 0.0 (e.g.
`test_imhd_briowu_2d`, the Maxwell PEC cavity unexcited components).
A change to the order of accumulations in the volume term can break
that. If we have to relax those tests we want to know exactly why.

---

## Task 2 — Cut register pressure in the DG core  ⚠️ PARTIAL — LANDED 2026-05-17

Single-accumulator pattern landed on `dg_core_3d.mojo`. `dfdx`, `dgdy`,
`dhdz`, and `surface` are all folded into one running `accum` array
(was 4·NC + others; now 1·NC). Same change was tried on the 2D core
and reverted — it broke `test_euler_123_2d` (positivity preserver
couldn't recover from the FP-reordered near-vacuum trajectory) and
`test_imhd_briowu_2d_hlld` (y-drift bit-identity assertion). The 2D
speedup ceiling from Task 1 alone is already 2.0–2.2× of baseline,
so the upside of pushing harder didn't justify the regression risk.
3D had no analogous breakage.

3D results vs original baseline (RTX 3090):
- `euler_3d`:    30.2 → 44.85 steps/s (**1.49×**)
- `imhd_3d`:     22.9 → 32.87 steps/s (**1.44×**)
- `imhd_glm_3d`: 24.6 → 31.61 steps/s (**1.29×**, just below the
  1.3× target — GLM's 9-component state is the worst case for
  register pressure)

`advection_3d` is essentially unchanged because the advection 3D
wrapper (`spatial/dg_3d.mojo`) uses an inline standalone kernel rather
than `_dg_3d_compute_rhs_core` — refactoring it to use the shared
core is the natural follow-up.

The Task-2 plan's other levers (move `q_neg`/`q_pos`/`f_star` to
shared, `@noinline` on HLLD inner, etc.) remain unexplored. They
would mainly help squeeze the last bit on `imhd_glm_3d`.



**Goal.** Get the kernel under the 64- or 128-register bucket that
unblocks `Block Limit Registers ≥ 16`, doubling or quadrupling
theoretical occupancy from 16.67 % to 33 %–66 %. Combined with
Task 1's wider warps this is what turns the "SMs busy issuing" into
"SMs busy issuing useful work."

**Files.** Primarily `src/warpxm/spatial/dg_core_{1,2,3}d.mojo`. The
per-physics flux closures and `apps/*.mojo` static methods may also
need to be re-ordered so the compiler can drop intermediates earlier.

**Where the registers go today** (read of `dg_core_2d.mojo`):

| Quantity | Live range | Size (NC=9) |
|---|---|---|
| `q_n` | top to end | 9 doubles |
| `fx_n` | after `flux_x_fn` to surface writes | 9 doubles |
| `fy_n` | after `flux_y_fn` to surface writes | 9 doubles |
| `dfdx` | volume loop to write-back | 9 doubles |
| `dgdy` | volume loop to write-back | 9 doubles |
| `surface` | first surface branch to write-back | 9 doubles |
| `q_neg` / `q_pos` (×4 faces) | inside each surface branch | 9 doubles |
| `f_star` (×4 faces) | inside each surface branch | 9 doubles |

That's ≥ 6 × 9 = 54 doubles permanently live during the surface
section, plus another 18 inside each branch — north of 100 doubles
in flight. At 64-bit each that's > 800 bytes ≈ way past the
register-file budget per thread; spills are inevitable.

**Approach (pick what works after measurement).**

1. **Single accumulator pattern.** Replace `dfdx`, `dgdy`, `surface`
   with one `accum` array. Compute and add into it in stages:
   ```
   accum = 0
   accum += -d/dx · fx     (drop fx live range here)
   accum += -d/dy · fy     (drop fy live range here)
   accum += surface_left + surface_right + surface_bottom + surface_top
   write accum / jacobian   (or fold jacobian into each step)
   ```
   Each face contribution writes directly into `accum` instead of
   into a separate `surface` array. Should shave ~18 NC-doubles.
2. **Don't keep `fx_n` / `fy_n` after they hit shared memory.** They're
   already written to `f_shared`; reload from there if needed. The
   compiler is presumably keeping them in registers because of the
   later surface-branch uses of `fx_n` / `fy_n` in the divisor —
   change the surface code to read those back from `f_shared` (or
   recompute from `q_n` which is cheap).
3. **Move per-face intermediates onto the stack/shared.** `q_neg`,
   `q_pos`, `f_star` are each used in exactly one branch; move them
   to `stack_allocation[..., AddressSpace.GENERIC]` inside the branch
   scope so the compiler doesn't promote them. If that's not enough,
   stash them in shared memory using lane-id as the second index
   (cost: ~9·4·NC bytes/block — small).
4. **`@parameter` flux closures.** Confirm they actually fully inline.
   Right now `numerical_flux_static` for HLLD is a large function
   (~215 lines of permutation logic) — if inlining brings hundreds of
   live values into the caller it'll wreck regs. Try `@noinline` or a
   separate __device__-style helper that takes/returns InlineArrays
   so the compiler can register-rotate.

**Acceptance.**
- ncu reports `Registers Per Thread ≤ 128` (preferably ≤ 80).
- ncu reports `Theoretical Occupancy ≥ 33 %` and
  `Achieved Occupancy` within 5 % of theoretical.
- Combined with Task 1, `scripts/bench.sh` shows ≥ 2.5× `steps/s`
  on 2D, ≥ 2.0× on 3D, vs. the pre-Task-1 baseline.
- All 70 tests still pass.

**Risk / things to watch.**
- **Spill-to-local under shared-memory tricks.** Storing in shared
  memory works only if the access pattern doesn't introduce serialised
  bank conflicts. Run ncu's `MemoryWorkloadAnalysis` after each
  change — `shared_st_bank_conflicts` and `shared_ld_bank_conflicts`
  should stay near 0.
- **MIO short-scoreboard stalls** (currently 82 % of total stall) will
  *increase* if we move more state to shared without first widening
  the warp (Task 1). Land Task 1 first.
- **Numerical drift** in the order of accumulations. The "single
  accumulator" pattern changes the order of additions for the volume
  term and the surface contributions. Several tests rely on exact
  cancellation (e.g. extruded 1D inputs giving exactly-zero y-drift).
  May require either (a) accepting tighter-but-nonzero tolerances on a
  handful of tests, with justification, or (b) preserving the order
  by writing into a single `accum` in a fixed sequence.

**Validation plan.** Same as Task 1 — run physics tests in order of
sensitivity, bit-identity tests first.

---

## Task 3 — (Follow-up) Investigate the limiter kernel in 3D  ⚠️ INVESTIGATED, JUSTIFICATION 2026-05-17

Re-measured after Tasks 1 + 2 landed (RTX 3090, `imhd_3d` 32³ P=2):
- with limiter:    33.26 steps/s (30.07 ms/step)
- without limiter: 36.31 steps/s (27.54 ms/step)

The Moe-Rossmanith limiter accounts for **~8.4 % of step time**, up
from the ~8 % the task-plan brief estimated pre-Task-1+2. Hitting the
1.5× limiter-kernel speedup target would land **~2.8 % overall**
speedup — non-zero but well below the wins from Tasks 1 + 2.

**Where the limiter time actually goes** (from reading
`src/warpxm/adjusters/moe_rossmanith.mojo` after Tasks 1+2):

The two kernels (`_bounds_kernel`, `_apply_kernel`) use a
*one-thread-per-element* layout — block_dim = 256, each thread loops
over `(nodes_per_elem · NC)` global-memory reads to compute its
element's cell average + min/max bounds. That's 243 reads/element for
3D P=2 NC=9.

The block-size mismatch problem that motivated Task 1's element packing
**does not apply here** — 256 threads/block is 8 full warps, no
under-utilization. The block-packing recipe gives nothing.

The register-pressure problem from Task 2 **does not apply either** —
the bounds kernel's `comptime for c in range(NC)` is *outer*, so only
`{sum, min, max}` are live at a time per thread (3 scalars, not 3·NC).

**The actual bottleneck is global-memory coalescing.** Adjacent
warp threads handle adjacent elements, so when a warp reads the same
`(n, c)` slot across its 32 threads, the addresses are
`q[e*nodes_per_elem*NC + n*NC + c]` for e = 0..31 — stride
`nodes_per_elem · NC · 8 = 1944` bytes between threads. That's 32
uncoalesced 128-byte transactions per warp-read instead of 1.

Fixing this would require either:
1. **A global memory layout change** — store `q[n][c][e]` instead of
   `q[e][n][c]`. Massively invasive: every spatial/temporal kernel in
   the codebase indexes the current layout, and the new layout would
   *worsen* the DG core's element-private reads (which currently
   coalesce well within a single thread's loop). Net negative.
2. **Per-element block cooperation** — `nodes_per_elem` threads per
   block (243 for 3D P=2), each handling one `(n, c)` slot, then
   shared-memory tree-reduction to compute sum/min/max. Threads then
   read `q[e*243*NC + tid*NC + c]` for c in [0, NC) — across threads
   in a warp this is `q[e*243*NC + tid*NC + c]` for tid = 0..31 with
   stride NC·8 = 72 bytes. Still not perfectly coalesced (NC = 9 isn't
   a power of two), but 2 transactions per warp-read instead of 32.
   That's the ~6× memory-bandwidth win that would unlock the 1.5×
   limiter speedup. Implementation cost: significant (cooperative
   reduction kernel + per-block shared memory layout + write-back
   pattern change).

**Decision: skip the limiter rewrite for now.** The 2.8 % overall
ceiling doesn't justify the engineering cost compared to MPI halo
exchange (project's stated next major milestone) or the deferred
Vlasov-Maxwell deepenings. Revisit if the limiter becomes a
disproportionate fraction of step time once mesh sizes or the
adjuster catalog grows.

(Original task brief preserved below for context.)



**Why now.** After Tasks 1+2, the DG core's share of GPU time drops
toward ~70 %, and the Moe-Rossmanith limiter — already 8 % in 3D
today — becomes the second-largest cost in absolute terms. Worth a
look but not before.

**What to do.** Re-profile after Tasks 1+2 land. Apply the same
recipe — block-packing + register reduction — to
`src/warpxm/adjusters/moe_rossmanith.mojo` if it shows the same
shape (low occupancy, low DRAM). Expect a smaller absolute win
because the limiter does less work per launch.

**Acceptance.** Either a ≥ 1.5× limiter-kernel speedup or a written
justification under `tasks/dg-core-performance.md` ("we looked; it's
already well-tuned because X") so we don't keep relitigating it.

---

## Task 4 — (Follow-up) Reduce SSPRK3 device-to-device memcpy  ❌ ATTEMPTED, REVERTED 2026-05-17

Attempted to eliminate the `u_save ← u` per-step D2D copy on the 2D
stepper by passing `u` directly as the implicit `u^n` reference and
using a new `_ssprk3_combine_inplace_kernel` variant for the final
stage (since Mojo's aliasing checker rejects passing `u.buf` as both
u0 and output to the existing combine kernel). Result on the
post-Task-1+2 bench:
- `bench_advection_2d`: 2767 → 2727 steps/s (≈ -1 %)
- `bench_imhd_2d`: 159 → 152 steps/s (≈ -4 %)

The eliminated copy is correctly gone (87/87 tests pass) but the
in-place kernel variant is *slower* than the regular 4-arg combine —
either Mojo codegen-level memory fences around the aliased read/write,
or just a worse register-allocation outcome on the smaller 3-arg
kernel. The 5 % target the task plan called for is unreachable when
the savings (0.2 % of step time at current mesh sizes) get spent on
codegen overhead.

Reverted to the original 4-buffer SSPRK3 layout. Worth revisiting
once mesh size or `NC` grows enough that the D2D copy is a larger
fraction of step time — at that point a fused stage kernel (the more
ambitious version of the task plan) may pay off where the minimal
variant didn't.



**Status.** Today the SSPRK3 stepper does one full state-buffer
`memcpy(u_save ← u)` per step — 10.6 MB on the 2D MHD-GLM benchmark
×100 steps = 1.17 GB of D2D traffic per run. nsys ranks this at 0.2 %
of GPU time today (because the DG core dwarfs everything), so this is
**not** a top-3 task — but after Tasks 1+2 it may grow to 5–10 % of
total, and it's the kind of thing that's better fixed before it
becomes load-bearing.

**Approach.** SSPRK3 with adjusters could be rewritten as a single
fused kernel that combines the stages in-place without the explicit
`u_save` snapshot, using a single extra DG-state buffer instead of
four. Compare the diff to `src/warpxm/temporal/ssprk3_{1,2,3}d.mojo`
and `temporal/ssprk3.mojo` (which already has the shared combine
kernels).

**Acceptance.** ≥ 5 % `steps/s` improvement on the smooth benchmarks
(where the limiter doesn't dominate) and zero correctness regressions
on the full 70-test suite.

---

## Order of operations

```
Task 0 (harness)  →  Task 1 (block packing)  →  Task 2 (register cut)
                                                       ↓
                                                Task 3 (limiter)
                                                Task 4 (memcpy)
```

Tasks 1 and 2 are interdependent — landing 2 without 1 won't help
(more occupancy of small warps is still small warps); landing 1
without 2 only gets a fraction of the win. Both should be done
together, or at least in the same release.

## Headline expected outcome

If both Task 1 and Task 2 hit their acceptance bars, we go from
**16.67 % occupancy and 1.3 % DRAM** to roughly **50 % occupancy and
~20–40 % DRAM**, with `steps/s` up **2–3×** on the representative
benchmarks. That puts the Mojo port within striking distance of
upstream Kokkos WARPXM's per-GPU throughput, which is the explicit
goal of the project.
