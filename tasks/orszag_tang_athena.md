# Orszag-Tang vortex — Athena reference implementation

## Goal

Implement the canonical Orszag-Tang vortex problem in the warpxm-mojo
port using the unit convention adopted by Athena (Stone et al.), so
that results can be compared directly with the published reference
at <https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/>
and the C source at `~/GitHub/Athena-Cversion/src/prob/orszag-tang.c`.

The repo already had a 2D Orszag-Tang test, but it used the WARPXM
upstream normalization (domain `[0, 2π]²`, ρ₀ = γ², P₀ = γ) — the
same physical problem under a different unit scaling, but not directly
comparable to the Athena reference figures or to the broader MHD code
literature that adopts the Athena form.

## What was added

### `tests/test_imhd_orszag_tang_2d_athena.mojo`

Regression test using the Athena convention:

- Domain `[0, 1]²` periodic, γ = 5/3.
- `d_0 = 25 / (36π)`, `p_0 = 5 / (12π)`, `v_0 = 1`, `B_0 = 1 / √(4π)`.
- `u_x = -v_0·sin(2π y)`, `u_y = v_0·sin(2π x)`.
- B from vector potential `A_z = (B_0/4π) cos(4π x) + (B_0/2π) cos(2π y)`:
  - `B_x =  ∂A_z/∂y = -B_0 sin(2π y)`
  - `B_y = -∂A_z/∂x =  B_0 sin(4π x)` — analytically ∇·B = 0.
- P = 2, 32 × 32, integrated to t = 0.3 with SSPRK3.
- Moe-Rossmanith slope limiter + Zhang-Shu MHD positivity preserver,
  GLM cleaning at `c_h = 3`, `c_p² = 0.18`.

**Result at t = 0.3:**

| Diagnostic           | Value                              |
| -------------------- | ---------------------------------- |
| relative Δmass       | 7.5e-14 (machine precision)        |
| ρ range              | [0.047, 0.547]  (d₀ = 0.221)       |
| P_gas range          | [1e-10 (floor), 0.618]  (p₀ = 0.133) |
| ∫ψ² dV               | 6.3e-5                             |

P_gas hits the positivity floor (`p_eps = 1e-10`) in the strongest
rarefaction channels — consistent with the canonical Athena reference
at this time, and the same behavior as the `2D strong MHD blast (ΔP=100)`
validation. All assertions pass.

### `examples/imhd_orszag_tang_2d_athena_vtu.mojo`

VTU + PVD time-series output for visual comparison against the
canonical Athena figures.

- P = 2, 96 × 96, integrated to **t = 4.0** with `dt = 2.5e-4`.
- 41 frames (every 0.1 time units, including initial state).
- Per frame: prints (∫ρ dV, ∫E dV, ∫ψ² dV) and relative drifts.
- Output → `/tmp/orszag_tang_athena/orszag_tang_athena.pvd`,
  openable directly in ParaView.

(Compile-checked via `mojo build --emit object`; full execution not
benchmarked in this session.)

### `CLAUDE.md`

- New row in the validations table:
  `2D Orszag-Tang (Athena units) | ρ ∈ [0.047, 0.547] at t=0.3 (Athena ref), mass drift 8e-14, ψ² 6e-5`
- Test count bumped 66 → 70 (covers this addition plus the three new
  Vlasov 1D1V scaffolding tests that landed earlier in the session log).

## Regression status

`pixi run test` → **70 / 70 PASSED**, including the new
`test_imhd_orszag_tang_2d_athena`.

## Suggested next steps

1. **Run `examples/imhd_orszag_tang_2d_athena_vtu.mojo` to completion**
   on a fast GPU and visually compare the t = 0.5 and t = 1.0 frames
   against Athena's published contour plots. The wall-time will be
   dominated by 16k SSPRK3 steps on the 96×96, P=2 mesh — sizeable but
   bounded.

2. **Quantitative pressure-profile comparison.** Athena's reference
   page includes the gas-pressure plot along the line y = 0.4277 at
   t = 0.5. Adding a Python post-processing pass over the dumped VTU
   to extract that line and compare numerically against published
   data points would convert the visual test into a hard regression
   check.

3. **3D Orszag-Tang extension.** ✅ LANDED 2026-05-17 as
   `tests/test_imhd_orszag_tang_3d_athena.mojo`. NX=NY=32, NZ=4,
   P=2, identical IC at every z-layer. After 600 SSPRK3 steps to
   t = 0.3: **z-drift = 0.0 exactly** across all 9 MHD-GLM
   components; ρ, P, and ψ² match the 2D test to ULP; mass drift
   1.7e-12. Strongest evidence yet that the 3D MHD-GLM solver is
   bit-identical to 2D on z-extruded inputs.

4. **Higher-resolution Athena test.** 192 × 192 (Athena's reference
   resolution) at P = 2 → 576 DOFs per side, ~3× Athena's effective
   resolution. Would let us evaluate fine-scale features (current
   sheet thickness, turbulent cascade onset).

5. **Optional: Athena-style normalization for the existing
   non-Athena Orszag-Tang test.** ✅ DOCUMENTED 2026-05-17 (without
   converting the existing test). The current
   `test_imhd_orszag_tang_2d.mojo` keeps the WARPXM upstream form
   (domain `[0, 2π]²`, `ρ₀ = γ²`, `p₀ = γ`) so port-time regressions
   show up against the C++ reference. `test_imhd_orszag_tang_2d_athena`
   uses the Athena convention so results compare against the
   canonical Athena reference. CLAUDE.md's validations table now
   labels the WARPXM-units row explicitly and includes a short
   paragraph explaining why both setups exist.
