# Features

What is implemented, by physics and dimension. Every checked cell has an
end-to-end regression test passing on the GPU; the quantitative results are
on the [Validation](validation.md) page.

All cells with a check have an end-to-end regression test passing on the GPU.

|                            |  1D | 2D  | 3D  |
| -------------------------- | :-: | :-: | :-: |
| Advection                  | ✅  | ✅  | ✅  |
| 5-moment Euler             | ✅  | ✅  | ✅  |
| Maxwell                    | ✅  | ✅  | ✅  |
| Single-fluid plasma        | ✅  | ✅  | ✅  |
| Two-species plasma         | ✅  | ✅  | ✅  |
| Ideal MHD                  | —   | ✅  | ✅  |
| MHD + Dedner GLM cleaning  | —   | ✅  | ✅  |
| Moe-Rossmanith limiter     | ✅  | ✅  | ✅  |
| ParaView VTU + .pvd output | —   | ✅  | ✅  |
| Non-periodic BCs           | ✅  | ✅  | ✅  |
| CFL time-step selection    | ✅  | ✅  | ✅  |
| Multi-rank (MPI) runs      | ✅  | ✅  | ✅  |

Supporting infrastructure:

- **BCs**: `BC_PERIODIC` (implicit), `BC_COPYOUT`, `BC_WALL_FREESLIP`,
  `BC_PEC_WALL` (Maxwell, ideal MHD 2D/3D), `BC_PMC_WALL` (Maxwell),
  `BC_VLASOV_OUTFLOW` (velocity-space boundaries), `BC_VLASOV_REFLECT`
  (specular wall for the kinetic `PhaseMesh`). GLM-MHD treats every
  kind as copy-out. Per-axis selection in the mesh constructor.
- **GLL polynomial orders**: P = 1, 2, 3, 4 tables; convergence rates of
  P+1 verified for P=1, 2, 3 on the 1D advection convergence test.
- **Time integration**: 3-stage Shu-Osher SSPRK3 and adaptive
  Dormand-Prince RK45, both device-resident and generic over the spatial
  dimension. Fixed-step (`run_until`) and CFL-driven (`run_until_cfl` with
  `solver.suggest_dt`) drivers.
- **Sources**: gravity (Euler), Lorentz force (Maxwell → fluid), current
  density (fluid → Maxwell), parabolic ψ damping (GLM-MHD). Reusable
  generic-NC infrastructure.

- **Several GPUs**: one MPI rank per GPU, Cartesian domain decomposition
  with one layer of ghost elements, for every physics above and for the
  kinetic solver (configuration space is decomposed, velocity space is
  not). Results are bitwise independent of the number of ranks
  (`pixi run test-mpi`). See
  [Running on several GPUs](concepts/parallel.md).

## Kinetic Vlasov-Maxwell

Phase-space DG (continuum kinetic), separate from the fluid matrix above.

|                                   | 1D1V | 1D2V | 1D3V | 2D2V |
| --------------------------------- | :--: | :--: | :--: | :--: |
| Vlasov free streaming             | ✅   | ✅   | ✅   | ✅   |
| Lorentz force (E + v × B)         | ✅   | ✅   | ✅   | ✅   |
| Vlasov-Maxwell (self-consistent)  | ✅   | ✅   | ✅   | ✅   |
| Moments (density, current, energy)| ✅   | ✅   | ✅   | ✅   |
| CFL step (`suggest_dt`)           | ✅   | ✅   | ✅   | ✅   |
| BGK collision operator            | ✅   | ◑    | ◑    | ◑    |
| 5-moment → kinetic initialisation | ✅   | ◑    | ◑    | ◑    |
| Velocity-space outflow BC         | ✅   | ✅   | ✅   | ✅   |
| Specular (reflecting) wall        | ◑    | ◑    | ✅   | ✅   |

◑: the code is generic over the dimensions, but there is no test at
that one.

Every column comes from one dimension-generic stack
(`PhaseMesh[XDIM, VDIM, P_x, P_v]` and friends). The same code runs
2D3V and 3D3V, where the suite checks the right-hand side (exact to
round-off on polynomial data), free streaming, and a coupled plasma
oscillation with all three current components.

The generic kernel runs one phase element per GPU block, so
`(P_x + 1)^XDIM (P_v + 1)^VDIM` cannot exceed the 1024 threads of a
block (a compile-time error otherwise). That caps the polynomial order
by dimension:

| Phase space       | 1D1V–1D3V, 2D2V | 2D3V | 3D3V |
| ----------------- | :-------------: | :--: | :--: |
| Largest `P`       | 4 (the basis limit) | 3 (exactly 1024) | 2 |

The specular wall is tested in free streaming, in prescribed electric
and magnetic fields, and in a coupled run with conducting walls on the
field solver (all 1D3V; free streaming also 2D2V).

Sources: `src/warpxm/spatial/kinetic/`, `mesh/phase_mesh.mojo`,
`variable/phase_state.mojo`, `temporal/ssprk3_phase.mojo` and
`ssprk3_vlasov_maxwell.mojo`. See
[Vlasov-Maxwell](physics/vlasov.md).

## Capability notes

- **Primitive-bound limiting** — Zhang-Shu positivity preservers for
  Euler (1D / 2D / 3D) and ideal MHD (2D / 3D). MHD preserver also
  works with GLM-MHD (NC=9 ψ component shrunk along with the rest).
- **Slope limiter tolerance** — the limiter only acts on deviations
  larger than `α = alpha_coeff · dx^1.5`. The default `alpha_coeff = 1000`
  (upstream's) gives α ≈ 0.35 on a 200-cell unit domain, so on problems
  whose solution is much smaller than that (near-vacuum, coarse meshes)
  it is a no-op and the positivity preserver is left to do all the work.
  The Einfeldt 1-2-3 tests pass `alpha_coeff=10` for that reason. Pick α
  below the smallest solution scale you need limited.
- **Slope limiter** — Moe-Rossmanith conserved-variable limiter,
  generic over the dimension and NC-parameterised
  (`MoeRossmanith[NDIM, P, NC]` in `adjusters/moe_rossmanith.mojo`).
- **Adaptive RK45 (Dormand-Prince)** — NC-parameterised generic
  steppers for all three dimensions
  (`DormandPrince45[NDIM, P, NC]`). Fixed-dt `step` and adaptive
  `step_adaptive` (embedded 4th-order estimate, dt clipped to
  [0.2·dt, 5·dt], atol/rtol scaled RMS norm). On smooth problems
  takes 6–40× fewer steps than fixed-dt SSPRK3. Per-step error norm
  is reduced fully on the GPU via a single-block tree-reduction
  kernel (`_dp45_sum_sq_kernel` in `temporal/dormand_prince.mojo`), so only one scalar lands on the host per
  step — no `map_to_host` of the full state buffers.
- **HLLD numerical flux** — shipped on `DGImhdSolver[NDIM, P]` and
  `DGImhdGLMSolver[NDIM, P]` (`USE_HLLD=True`, Miyoshi-Kusano 2005
  five-state solver). Permutation-based core
  (`_hlld_core_3d` in `apps/imhd.mojo`) handles all three
  axis-aligned face directions in ~215 lines. Default path stays
  Rusanov (no regression). GLM-MHD path composes IMHD HLLD with the
  exact (B_n, ψ) 2x2 Riemann block (Dedner et al. 2002 §3.2;
  Mignone & Tzeferacos 2010 §4).
- **Pair-plasma diagnostic** — the 1.85e-3 antisymmetry residual on
  the 2D pair-plasma test was originally listed as a Strang-splitting
  motivation. `tests/test_pair_plasma_asym_scaling_2d.mojo` shows the
  residual scales as δ² (ratio 100 between successive δ values),
  confirming it's the *physical* O(δ²) coupling from the ρ-dependent
  Lorentz force `F = α·ρ·E` — species 1 has ρ ≈ 1+δ·sin, species 2
  has ρ ≈ 1−δ·sin, so their force amplitudes differ by 2·δ². No
  operator-splitting fix can eliminate it; closed as "physical, not
  algorithmic."

## Open work

Feature-matrix-complete, apart from 1D MHD (an intentional gap: ∇·B = 0
makes it degenerate). What remains is scale:

1. **Multi-GPU scaling**: the halo exchange is in place (see below) but
   has only been run with ranks sharing one GPU. Open are a measured
   weak-scaling curve (`bench/bench_mpi_weak_imhd_glm_3d.mojo`), a run
   with GPU-aware MPI, and overlapping the exchange with the interior
   right-hand side if the curve calls for it.
2. **Kinetic solver**: still open are curvilinear (cylindrical)
   configuration coordinates, positivity preservation for `f`, and a
   second species.
