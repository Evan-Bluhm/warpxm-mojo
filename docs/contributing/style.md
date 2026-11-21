# Mojo style notes

Conventions used throughout the repo, plus the most common pretraining-
era Mojo syntax mistakes to watch for.

## Pretraining-era pitfalls

Mojo's syntax has evolved fast. These are the patterns models from
mid-2024 still emit that **don't work** in this codebase:

| Wrong | Right |
| ----- | ----- |
| `alias N = 32` | `comptime N = 32` |
| `fn step(...)` | `def step(...) raises` |
| `inout self` | `mut self` (or `out self` in `__init__`) |
| `@parameter if cond:` | `comptime if cond:` |
| `@parameter for i in range(N):` | `comptime for i in range(N):` |
| `let x = 5` | `var x = 5` |
| `borrowed x: Int` / `read x: Int` | `imm x: Int` (usually implicit, rarely written) |
| `owned x: List[Int]` | `var x: List[Int]` (in arg lists) |
| `List[T](a, b, c)` | bracket literal `[a, b, c]` |
| `from collections import List` | `from std.collections import List` |
| `T` (bare param inside its struct) | `Self.T` |
| `constrained(cond, msg)` | `comptime assert cond, msg` |
| `InlineArray[T, N]` | `Array[T, N]` — no longer implicitly copyable: `return out^` for locals, `q.copy()` for borrowed args |
| `from std.gpu import …` | `from max.gpu import …` (ships in the `max-core` package) |
| `@parameter def f(...)` passed as `step[f](...)` | `def f(...) raises {mut limiter, imm}:` passed as a trailing argument, `step(..., f)`; name `mut` only for what the closure mutates |
| `row_major(Idx(n))` | `row_major(Int(n))` |
| `n: Int` as a GPU-kernel argument | `n_i64: Int64`, then `var n = Int(n_i64)` in the body; pass `Int64(n)` at the launch site (`Int` is not `DevicePassable`) |
| `comptime if P == 1: … elif P == 2: …` chains | `comptime match P:` with `case 1:` / `case _:` |

When in doubt: search the existing source for the construct you're
about to write. If it doesn't appear there, you're probably about to
hit a parse error.

## In-repo conventions

### File names

* Spatial solvers, steppers, adjusters, writers all end in
  `_1d.mojo` / `_2d.mojo` / `_3d.mojo`. Apps (`advection.mojo`,
  `euler.mojo`, `maxwell.mojo`, `imhd.mojo`, `imhd_glm.mojo`) are
  single canonical files — the DG launcher owns dimensionality.
  Dimensionless support modules (basis, core, the SSPRK3 combine
  kernels in `ssprk3.mojo`) also drop the suffix. The DP45 family
  has no standalone dimensionless file: its shared combine /
  error / reduction kernels live inside `dormand_prince_1d.mojo`
  and are imported from there by the 2D and 3D wrappers.
* Underscored prefixes mark "internal helpers": `_load_node`,
  `_combine1`, `_apply_bc_imhd_3d`, etc. These aren't expected to be
  called from outside the file.

### Comptime parameter order on structs

Polynomial order first, then component count, then flags:

```mojo
struct DGImhdSolver2D[P: Int, USE_HLLD: Bool = False]: ...
struct PositivityMHD2D[P: Int, NC: Int]: ...
```

The convention exists so multiple structs combine without parameter-
reorder confusion when used together.

### Real type and math

* All physics scalars are `Real` (= `Scalar[DType.float64]`). Never
  use bare `Float32` or `Float64` in physics code.
* Always import `sqrt`, `fabs`, `fmin`, `fmax` from
  `warpxm.core.math_compat`. The standard `std.math.sqrt` doesn't
  cover `float64` GPU paths on NVIDIA.

### Kernels never raise

GPU kernel functions (the `def ..._kernel_*d[...](...)` definitions)
**don't** have `raises`. The Mojo GPU dispatch path doesn't support
exceptions.

### Host driver code does raise

`def main() raises:` is the standard signature for every example and
test. Any host-side function that issues `enqueue_*` or `map_to_host`
calls needs `raises`.

### Don't write comments that restate the code

If you can read the identifier names and tell what's happening, no
comment is needed. Comments are for non-obvious WHYs:

* a hidden invariant ("`u_save` exists to break aliasing"),
* a workaround for a known issue ("NVIDIA `std.math.sqrt` missing on
  float64"),
* a subtle algorithmic detail ("e_i are b5 − b4, with e2 = 0").

Don't comment `for i in range(NX):` with `# Loop over X`.

### Don't write multi-paragraph docstrings

A short top-of-file comment explaining what's in the file is fine.
Per-function docstrings should be one short sentence. The longer
treatments live in [`docs/`](../).

## Reach for the skills

The repo bundles two Claude-compatible skills:

* `mojo-syntax` — current Mojo syntax reference.
* `mojo-gpu-fundamentals` — GPU programming in Mojo, indexed for the
  patterns you'll need (TileTensor, kernel launch, comptime
  specialisation).

These get loaded automatically when working in this repo via
`.claude/skills/`. They cost a bit of context window but save a lot of
syntax-mistake-hunting time.
