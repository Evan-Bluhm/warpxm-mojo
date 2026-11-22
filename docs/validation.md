# Validation

Quantitative results from the regression suite (`tests/`, one standalone
program per test). Run it with `pixi run test`; see
[Tests](contributing/tests.md) for how the suite is organised.

| Test                            | Quantitative result                                     |
| ------------------------------- | ------------------------------------------------------- |
| 1D advection convergence        | observed rates 1.98 / 3.01 / 3.99 (theoretical P+1)     |
| 2D advection convergence        | observed rates 1.90 / 3.03 / 4.00 (theoretical P+1)     |
| 3D advection convergence        | observed rates 3.12 / 3.99 at P=2,3 (theoretical P+1)   |
| Advection, odd element counts   | 1D / 2D / 3D meshes that leave a partial kernel block: max error at the discretisation level, tracer integral to 3e-14 |
| 1D Maxwell plane-wave conv      | observed rates 1.98 / 3.01 / 4.00 (theoretical P+1)     |
| 1D Euler entropy-wave conv      | observed rates 1.99 / 2.95 / 4.00 (theoretical P+1)     |
| 2D MHD Alfvén-wave conv         | observed rates 1.98 / 2.99 / 4.00 (theoretical P+1)     |
| 3D MHD Alfvén-wave conv         | observed rates 2.98 / 4.02 at P=2,3 (theoretical P+1)   |
| 1D1V Vlasov free-streaming conv | observed rates 1.78 / 3.10 / 4.29 (theoretical P+1)     |
| 1D2V Vlasov free-streaming conv | observed rates 1.97 / 3.00 / 3.99 (theoretical P+1)     |
| 2D Euler hydrostatic balance    | \|v\| ~ 4e-16 after 50 steps (well-balanced source)     |
| 2D Sod tube                     | analytic plateaus match to 3e-3                         |
| 1D Langmuir (single-fluid)      | mass drift 2.73e-13, ρ at sample within 7e-6 of IC after period |
| 1D pair-plasma Langmuir         | ρ values match 2D-extruded result to 14 sig figs; mass drift 1.9e-13 per species |
| 2D pair-plasma Langmuir         | mass conserved per species to machine epsilon           |
| Pair-plasma asym δ-scaling      | ratio 100.0 / 100.0 between δ ∈ {0.01, 0.001, 0.0001} — physical O(δ²) |
| 2D Maxwell PEC cavity TM₁₁      | unexcited components stay at 0.0 exactly                |
| 2D MHD Brio-Wu                  | flux balance of all conserved integrals to 2e-12; rarefactions isentropic to 3e-4 |
| 2D Orszag-Tang (upstream WARPXM units) | mass drift 6e-14, ψ² bounded — domain [0, 2π]², ρ₀=γ², p₀=γ |
| 2D Orszag-Tang (Athena units)   | ρ ∈ [0.047, 0.547] at t=0.3 (Athena ref), mass drift 8e-14, ψ² 6e-5 |
| 3D Orszag-Tang (Athena, z-extruded) | z-drift 0.0 exactly; ρ + P + ψ² match 2D bit-for-bit, mass drift 1.7e-12 |
| 2D GLM ψ-bump decay             | ~900× decay over 2 time units (45000× over 4)           |
| 3D MHD blast                    | ∫B_x dV preserved to 4e-15 (Faraday's law)              |
| 3D cyclotron orbit              | ΔE = 3e-13 (magnetic force does no work)                |
| 1D Einfeldt 1-2-3 problem       | L1(ρ) 7.3e-3 vs exact; centre ρ 0.025 (exact 0.0219); flux balance 1e-14 |
| 2D Einfeldt 1-2-3 problem       | y-extruded 1D; same L1(ρ) 7.3e-3; transverse momentum 4e-15; mass 1e-14 |
| 3D Einfeldt 1-2-3 problem       | y/z-extruded; matches the 2D result to 12 digits        |
| 2D strong MHD blast (ΔP=100)    | P_gas clamped to p_eps; mass / ∫B_x preserved 2e-13    |
| 3D strong MHD blast (ΔP=100)    | P_gas clamped to p_eps; mass / ∫B_x preserved 2e-13    |
| 2D Brio-Wu (HLLD)               | same flux-balance and isentropy checks as the Rusanov path |
| 3D Brio-Wu (HLLD)               | exact y/z-extrusion (drifts 0), same checks as Rusanov  |
| Adaptive DP45 (smooth advection)| 186 steps (vs ~2000 SSPRK3), L2 3.1e-7 at rtol 1e-5; error tracks tolerance |
| Temporal order (1D Euler)       | SSPRK3 2.99, DP45 5.00 against a small-step reference  |
| HLLD Alfvén-wave convergence    | rate 3.00 at P=2; x / y / z propagation agree to round-off |
| MHD wall BCs (closed box)       | mass 6e-15; PEC conserves energy to 8e-15, free-slip does not |
| Maxwell PMC cavity              | dual of the PEC cavity; identical errors in 1D/2D/3D   |
| Sod / Lax shock tubes           | L1 vs exact Riemann solution 5e-3..1e-2; flux balance 1e-13 |
| Euler positivity preserver      | cell means to 1e-15; worst node lands on the floor to 1e-12 |
| CFL time-step selection         | `suggest_dt` matches the analytic bound to 1e-13 for every physics; SSPRK3 advection stable at cfl 1, unstable at cfl 4 (P=1..4) |
| CFL-driven vs fixed-dt run      | identical to 4e-15 on a 1D Euler entropy wave; lands on the target time exactly |
| Mesh helpers (fill / integrate) | match the mesh coordinate buffers and closed-form integrals to 1e-13 in 1D/2D/3D, 1D1V, 1D2V |
| VTU writer (appended binary)    | every array round-trips to 1e-15; file layout checked block by block |
| Vlasov-Maxwell uniform oscillator | E and J match E0 cos t, −E0 sin t to 2e-6 each quarter period |
| Vlasov-Maxwell Weibel growth    | fitted γ 0.36445 vs kinetic theory 0.36436             |
| Vlasov-Maxwell Landau-damped EM wave | period within 5e-4, damping within 5e-3 of kinetic root |
| Adaptive DP45 (2D entropy wave) | 200 steps for t=1, dt 5.3e-4..1e-2, ρ L2 6.6e-7        |
| Adaptive DP45 (2D MHD Alfvén)   | 169 steps for t=1 (vs 1000 SSPRK3), L2 ~4.3e-6 on B/u  |
| 1D Maxwell PEC cavity TM        | L2 E_z 8e-6 after one period, unexcited components 0.0 |
| Adaptive DP45 (1D Maxwell)      | 52 steps for t=1 (vs ~2000 SSPRK3), L2 1e-4, transv 0.0 |
| Adaptive DP45 (2D Maxwell)      | 76 steps for t=1/√2 (vs ~1414 SSPRK3), L2 ~1e-4 on E/B |
| Adaptive DP45 (3D advection)    | 64 steps for t=1, dt 0.008..0.016, L2 1.76e-3          |
| Adaptive DP45 (3D MHD Alfvén)   | 131 steps for t=1 (vs 1000 SSPRK3), L2 ~2.4e-5 on B/u  |
| Adaptive DP45 (3D Maxwell)      | 50 steps for t=1 (vs ~2000 SSPRK3), L2 ~8e-4 on E/B    |

**Two Orszag-Tang setups, on purpose.** `test_imhd_orszag_tang_2d`
uses the upstream WARPXM convention (domain `[0, 2π]²`, `ρ₀ = γ²`,
`p₀ = γ`) so any port-time regression shows up against the C++
reference. `test_imhd_orszag_tang_2d_athena` uses the Athena
convention (domain `[0, 1]²`, `ρ₀ = 25/(36π)`, `p₀ = 5/(12π)`,
`v₀ = 1`, `B₀ = 1/√(4π)`) so results compare directly with the
canonical reference at
<https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/>
and the broader MHD-code literature that adopts Athena's form. The
3D variant only exists in Athena units (z-extrusion of the Athena
2D IC).
