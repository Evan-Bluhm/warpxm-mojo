# Vlasov-Maxwell port — next steps

## Current state (after slice 22)

The kinetic Vlasov-Maxwell solver is functional in **1D1V** and end-to-end
validated. **1D2V** is feature-complete for uncoupled and coupled
operation. The Weibel instability infrastructure is in place and a
characterisation test runs cleanly with the analytic linear-eigenmode
IC — but the observed `γ_obs` does **not** match Davidson's theory
prediction at this mesh / k regime (peak `max|B_z|` is only ≈ 1.1× the
seed before decaying). Tightening the Weibel reproduction is the
natural next deepening pass — see "Open Weibel deepening" below.
Twenty-two accumulated slices since 2026-05-16, all uncommitted in WIP
form (`git status` shows ~30 untracked files under
`src/warpxm/{apps,mesh,spatial,temporal,variable}/` and `tests/`):

| Slice | What landed                                                               |
| ----- | ------------------------------------------------------------------------- |
| 1     | `PhaseState1D1V`, `PhaseMesh1D1V` (geometry), `VlasovMaxwell` app skeleton |
| 2     | Face connectivity + flux helpers (`internal_flux_x/v_static`, upwind `numerical_flux_static`) |
| 3     | DG kernel `dg_vlasov_1d1v.mojo`, `SSPRK3_1D1V` stepper, free-streaming test (L²=6.4e-4) |
| 4     | Maxwell aux coupling — `av = (q/m)/λ · E_x` per phase node                |
| 5     | Zeroth-moment kernel `ρ = ∫f dv`                                          |
| 6     | First-moment kernel `J = ∫v f dv`                                         |
| 7     | Vlasov→Maxwell current source `apply_vlasov_current_source_1d`            |
| 8     | Coupled SSPRK3 stepper, uniform oscillator (period closes to 7e-8)        |
| 9     | Spatial Langmuir mode + energy/mass conservation diagnostics (ΔW/W = 8e-12) |
| 10    | `BC_VLASOV_OUTFLOW` at v-boundaries (replaces COPYOUT fallback)           |
| 11    | Second-moment kernel `E_kin = ½∫v² f dv` (slice-11 pivot from Bohm-Gross) |
| 12    | BGK collision source `vlasov_bgk_source_1d1v.mojo` (Maxwellian fixed-point at 3e-8) |
| 13    | BGK relaxation dynamic test (counter-streaming IC ⇒ Maxwellian, L²=7.5e-6)|
| 14    | `apply_5moment_to_kinetic_1d1v` initialiser (mirrors upstream `5moment_to_kinetic.{h,cc}`) |
| 15    | Quantitative Bohm-Gross dispersion (period err vs theory −1.3 %)          |
| 16    | Hybrid pipeline test: fluid IC → kinetic initialiser → coupled stepper    |
| 17    | 1D2V scaffolding (state + geometry mesh + smoke test)                     |
| 18    | 1D2V face connectivity (six-direction, BC_VLASOV_OUTFLOW at v-edges) + Lorentz helpers `acceleration_v1/v2_1d2v_static` + flux aliases |
| 19    | 1D2V DG spatial kernel `dg_vlasov_1d2v.mojo` (volume + 6 surface branches) + `DGVlasovSolver1D2V` launcher; free-streaming smoke test (L²=3.5e-4 after 200 forward-Euler steps at dt=5e-4) |
| 20    | 1D2V moment kernels (zeroth / J_x / J_y / E_kin) in `vlasov_moments_1d2v.mojo` + `SSPRK3_1D2V[P_x, P_v1, P_v2, NC]` stepper; 2D Maxwellian validation (ρ err 1.6e-8, E_kin err 1.4e-9, J_x/J_y at machine ε) |
| 21    | 2-component Vlasov→Maxwell current source `apply_vlasov_current_source_1d2v` (J_x → E_x, J_y → E_y) + `SSPRK3_VlasovMaxwell_1D2V` joint stepper; coupled-stepper smoke test on bi-Maxwellian / B_z-seed setup (mass drift 6e-11) |
| 22    | Weibel characterisation: analytic linear-Vlasov eigenmode IC at Davidson quartic-corrected γ_W ≈ 0.69 for σ_par²=0.25 / σ_perp²=1 / L=4 / k=π/2. Mass drift 1.2e-10, peak `max\|B_z\|` ≈ 1.1× δ, no blow-up. γ_obs/γ_W mismatch documented (eigenmode IC needs full-Z(ζ) root for accuracy) |

**Test suite: 87 / 87 passing.** All slices are accumulated WIP — the user
lands focused commits manually after review (recent commits in `git log`
show that pattern: `d80914d`, `09ef49b`, `63db39c`, etc. landed
slice-1-through-9 doc/code in clean focused units).

## Open Weibel deepening

The slice-22 characterisation test ran the analytic linear eigenmode
IC and observed `peak |B_z| ≈ 1.1 · δ` followed by decay. Mass and
amplitude diagnostics are clean — what's missing is a high-fidelity
eigenmode IC that pushes essentially all amplitude into the growing
branch. Two known issues drive the mismatch:

1. **Davidson's quartic is only the `ζ → ∞` limit of the kinetic
   dispersion.** At our regime `γ/(k v_t_par) ≈ 2.6`, the next
   term `ζZ(ζ) → -1 − 1/(2ζ²) − 3/(4ζ⁴) − …` is not yet asymptotic.
   The eigenvalue solver needs to root-find the full `ζ Z(ζ)`
   relation with `Z(ζ)` the complex plasma dispersion function.
   That is a sizeable arithmetic-stack addition (the Faddeeva /
   `w(z)` function).
2. **Eigenmode IC requires both branches' magnitudes to be set
   self-consistently.** Even with the right γ_W, the δf formula
   derived from `δf̂ = -α[…]/(γ + ikv1)` admits two roots at ±γ_W,
   so the IC superposes growing + decaying modes. Need to set the
   relative weight from a separate Maxwell-current self-consistency
   check (`J_y(x, 0) = -ε₀ · B_z_amp · (γ²+k²c²)/k · cos(kx)`
   must match `∫∫ v2 δf(x, v1, v2, 0) dv1 dv2`).

Either fix is a sizeable analysis/implementation slice — set aside
for a focused future pass. The slice-22 test passes as a
characterisation in the meantime.

## Concrete near-term slices

## Alternative directions (not 1D2V)

### A. Convergence study  ✅ LANDED 2026-05-17

`tests/test_vlasov_convergence_1d1v.mojo` runs free-streaming
`f₀ = sin(2π x) · exp(-v²/σ²)` for `P ∈ {1, 2, 3}` at `NX ∈ {32, 64}`,
`NV = 8`, `P_V = 3`, `T = 0.1`, `dt = 2·10⁻⁴` (pinned to the finest-
mesh CFL across the whole sweep so SSPRK3's `O(dt³)` doesn't pollute).

Measured rates (RTX 3090):

1D1V (`test_vlasov_convergence_1d1v.mojo`):
- `P = 1`: 1.78 (theoretical 2.0)
- `P = 2`: 3.10 (theoretical 3.0)
- `P = 3`: 4.29 (theoretical 4.0)

1D2V (`test_vlasov_convergence_1d2v.mojo` — same recipe extended to
the slice-19 kernel; `NV1 = NV2 = 4, P_V1 = P_V2 = 2`):
- `P = 1`: 1.97 (theoretical 2.0) — essentially exact
- `P = 2`: 3.00 (theoretical 3.0) — essentially exact
- `P = 3`: 3.99 (theoretical 4.0) — essentially exact

The Bohm-Gross variant the original brief mentioned remains as a
follow-up — would need careful fit to the dispersion relation
rather than the trivial free-streaming analytic, but the test
infrastructure here ports over directly.

### B. MPI halo exchange

Multi-GPU port. The configuration-space DG (`dg_core_{1,2,3}d.mojo`)
doesn't have MPI either, so this is a project-wide infrastructure
addition rather than Vlasov-specific. Probably the next major milestone
after the 1D2V Vlasov port is functionally complete.

### C. Performance baseline + optimisation

`nsys` profile the 1D1V Bohm-Gross run, identify hot kernels, see if
`[[dg-core-perf-ceiling]]` (16.67 % occupancy, 255 reg/thread) applies
to `_vlasov_rhs_kernel_1d1v`. Probably yes — same kernel structure as
`dg_core_2d.mojo` with `(P_x+1, P_v+1)` block size.

Defer unless the user explicitly redirects.

## Constraints carried forward across slices

These tolerance / parameter-choice lessons should be applied to every
slice that touches a Maxwellian:

1. **Maxwellian tail truncation is `erfc(v_max/(v_th·√2))`, not
   `exp(-v_max²/V_th²)`.** First is the integrated tail mass (~ 4-5
   orders of magnitude larger than the peak value at the boundary).
   Test tolerances must budget this. Discovered in slices 13 and 14.
2. **For tight mass conservation under active v-acceleration, size
   `v_max ≥ 8 · v_th`.** First attempt in slice 16 used `5.7·v_th` and
   leaked mass at `erfc(4) ≈ 1.5·10⁻⁵`; bumping to `10·v_th` recovered
   bit-precision conservation. Cumulative lesson from slices 13/14/16.
3. **`std.math.sqrt` doesn't lower for `DType.float64` on NVIDIA GPUs.**
   Use `warpxm.core.math_compat.sqrt` in any kernel-side code. Caught in
   slice 12; documented in `core/math_compat.mojo`.
4. **Capturing closures cannot be materialised as runtime function
   arguments.** Use `mut out: Real` parameters instead of returning
   `(Real, Real, Real)` tuples or passing weight functions as runtime
   arguments. Caught in slice 13.
5. **BGK "conserves moments by construction" is continuum-only.** With
   GLL quadrature on a truncated v-domain, BGK drifts moments by exactly
   the tail residue per timestep. Slice 13.
6. **The phase-state index layout is axis-N-outermost / axis-0-innermost
   (matches `dg_core_2d/3d.mojo`).** For 1D1V: `local_n = iv·(P_x+1) + ix`.
   For 1D2V: `local_n = (iv2·(P_v1+1) + iv1)·(P_x+1) + ix`. Carry through
   to any new kernel; slice 3 lost an iteration on flipping this
   convention.

## Open items in user-driven side work

Two new files in `git status` were authored by the user, not by /loop
slices:

- `examples/imhd_orszag_tang_2d_athena_vtu.mojo`
- `tests/test_imhd_orszag_tang_2d_athena.mojo`

These are unrelated to the Vlasov port; tracked in
`tasks/orszag_tang_athena.md` already.

There is also a memory note `memory/dg_core_perf_ceiling.md` indicating
the user has run `nsys`/`ncu` profiling on the configuration-space DG
core and identified an occupancy ceiling — referenced by alternative
direction C above.
