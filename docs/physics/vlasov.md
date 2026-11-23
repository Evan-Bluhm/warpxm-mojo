# Vlasov-Maxwell (kinetic)

Continuum-kinetic solver: the distribution function $f(x, \mathbf{v}, t)$
of one species is evolved on a phase-space DG mesh and coupled to
Maxwell's equations through its current.

There are two implementations:

* The **dimension-generic stack**, parameterised by `[XDIM, VDIM, P_x,
  P_v]`: `XDIM` configuration axes and `VDIM` velocity axes, `1 ≤ XDIM ≤
  VDIM ≤ 3`. It covers 1D1V, 1D2V, **1D3V**, 2D2V, 2D3V and 3D3V with one
  kernel. Use it for new work; see
  [Dimension-generic stack](#dimension-generic-stack).
* The original hand-written **1D1V** $(x, v_x)$ and **1D2V**
  $(x, v_x, v_y)$ types (suffix `1D1V` / `1D2V`), described in the rest
  of this page. BGK collisions and the fluid-to-kinetic initialiser exist
  only there. On the same data the generic and the hand-written
  right-hand sides agree to round-off
  (`test_vlasov_generic_rhs_1d`).

This is a separate code path from the fluid solvers: it has its own
meshes, state types, kernels and steppers. The Maxwell field lives on an
ordinary `BlockMesh[XDIM, P_x]` / `DGState[XDIM, P_x, MAXWELL_NC]`.

## Equations

$$
\partial_t f + \mathbf{v}\cdot\nabla_x f
+ \frac{q/m}{\lambda}\,(\mathbf{E} + \mathbf{v}\times\mathbf{B})\cdot\nabla_v f = 0 .
$$

Reduced to the implemented phase spaces:

| | x-flux | velocity-space acceleration |
| - | ------ | --------------------------- |
| 1D1V | $v\,f$ | $a_v = \dfrac{q/m}{\lambda} E_x$ |
| 1D2V | $v_1 f$ | $a_{v_1} = \dfrac{q/m}{\lambda}(E_x + v_2 B_z)$, $\; a_{v_2} = \dfrac{q/m}{\lambda}(E_y - v_1 B_z)$ |
| 1D3V | $v_1 f$ | $a_{v_1} = \dfrac{q/m}{\lambda}(E_x + v_2 B_z - v_3 B_y)$, $\; a_{v_2} = \dfrac{q/m}{\lambda}(E_y + v_3 B_x - v_1 B_z)$, $\; a_{v_3} = \dfrac{q/m}{\lambda}(E_z + v_1 B_y - v_2 B_x)$ |

With more configuration axes the x-flux gains $v_2 f$ along $y$ and
$v_3 f$ along $z$.

The flux is linear in $f$, so every face uses the upwind (Rusanov) flux
with the local coordinate or acceleration as the wave speed.

Maxwell receives the species current:

$$
\partial_t \mathbf{E} \mathrel{+}= -\frac{\alpha}{\varepsilon_0}\,\mathbf{J},
\qquad \mathbf{J}(x) = \int \mathbf{v}\, f\, d\mathbf{v},
$$

with $\alpha = q/m$. In 1D1V only $E_x$ is driven; in 1D2V both $E_x$
and $E_y$; with three velocity dimensions all of $\mathbf{E}$.

## Dimension-generic stack

```mojo
from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *

comptime Phase = PhaseState[1, 3, P_X, P_V, VLASOV_NC]     # 1D3V
comptime Fields = DGState[1, P_X, MAXWELL_NC]

# Axes are listed x first, then v.
var mesh = PhaseMesh[1, 3, P_X, P_V](
    ctx,
    lo=[X_MIN, -V_MAX, -V_MAX, -V_MAX],
    hi=[X_MAX, V_MAX, V_MAX, V_MAX],
    n=[NX, NV, NV, NV],
    periodic=[True],
)
var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=True)
var f = Phase(ctx, mesh.num_phase_elements)
var q_maxwell = Fields(ctx, NX)

def initial(x: Real, v1: Real, v2: Real, v3: Real) -> Real: ...
mesh.fill(f, initial)                    # also visit_nodes, integrate

var spatial_vlasov = DGVlasovSolver[1, 3, P_X, P_V](app)
var spatial_maxwell = DGMaxwellSolver[1, P_X](maxwell_app)
var moments = VlasovMoments[1, 3, P_X, P_V](ctx, mesh)
var mom = DGState[1, P_X, VLASOV_MOMENT_NC](ctx, NX)
var stepper = SSPRK3VlasovMaxwell[1, 3, P_X, P_V](
    ctx, mesh.num_phase_elements, mesh.num_x_elements,
)

def compute_joint_rhs(
    mut f_in: Phase, mut qm_in: Fields, mut f_out: Phase, mut qm_out: Fields,
) raises {mut mom, imm}:
    spatial_vlasov.compute_rhs(ctx, mesh, f_in, qm_in, f_out)
    spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
    moments.compute(ctx, mesh, f_in, mom)
    apply_vlasov_current_source[1, 3, P_X](ctx, mom, qm_out, ALPHA, EPS0)

def suggest_dt(mut f_: Phase, mut qm: Fields) raises {imm} -> Real:
    return min(
        spatial_vlasov.suggest_dt(ctx, mesh, qm), spatial_maxwell.suggest_dt(x_mesh),
    )

t = stepper.run_until_cfl(
    ctx, f, q_maxwell, t, t_next, compute_joint_rhs, suggest_dt,
)
```

* **Mesh.** `PhaseMesh[XDIM, VDIM, P_x, P_v]` has one order for the
  configuration axes and one for the velocity axes. A configuration axis
  is periodic or carries `BC_COPYOUT`, `BC_VLASOV_OUTFLOW` or
  `BC_VLASOV_REFLECT` per side (the five-argument constructor takes
  `bc=[...]`, two entries per axis); velocity boundaries are always
  `BC_VLASOV_OUTFLOW`. See [Walls](#walls). The mesh is
  structured, so it stores no face tables: the kernel finds neighbours by
  index arithmetic. `fill` and `visit_nodes` take a closure of the
  coordinates, x axes first: `ic(x, v1, v2, v3)` in 1D3V,
  `ic(x, y, v1, v2)` in 2D2V.
* **Solver.** `DGVlasovSolver[XDIM, VDIM, P_x, P_v]` reads all six field
  components the species can feel from a `DGState[XDIM, P_x, 6]` in
  Maxwell layout on the companion `BlockMesh[XDIM, P_x]`.
  `suggest_dt(ctx, mesh, fields)` bounds the step from the velocity extent and
  the largest field components, which are reduced on the device; it does
  not cover the field solver. Each call still waits for the device queue,
  so give `run_until_cfl` a `dt_interval` of 10 or 20 unless the fields
  change quickly.
  `mesh.check_companion(x_mesh)` raises if the `BlockMesh` holding the
  fields does not have the same extent, element counts and periodicity;
  call it once after building the two meshes.
* **Moments.** `VlasovMoments.compute` fills a
  `DGState[XDIM, P_x, VLASOV_MOMENT_NC]` with `MOM_DENSITY`,
  `MOM_FLUX_X`, `MOM_FLUX_Y`, `MOM_FLUX_Z` and `MOM_ENERGY`
  ($\tfrac12\int |v|^2 f$) in a single pass over `f`. Build it once
  with `VlasovMoments[...](ctx, mesh)`: it owns a small scratch buffer.
  The velocity elements are reduced in chunks of at most 16 per thread,
  so the pass costs about an eighth of a right-hand side whatever the
  split between configuration and velocity resolution.
  `apply_vlasov_current_source[XDIM, VDIM, P]` adds
  $-\alpha\,\Gamma_d/\varepsilon_0$ to the first `VDIM` components of
  $\mathbf{E}$.
* **Steppers.** `SSPRK3Phase[XDIM, VDIM, P_x, P_v, NC]` advances `f` in
  prescribed fields; `SSPRK3VlasovMaxwell[XDIM, VDIM, P_x, P_v]` advances
  `(f, E, B)`. Both have `step`, `run_until` and `run_until_cfl`, and
  both keep two scratch copies of each state (a stage value and the
  right-hand side), overwriting the stages in place.
* **Cost.** One phase element is one GPU block, so
  `(P_x + 1)^XDIM (P_v + 1)^VDIM ≤ 1024`: any `P` up to 4 through
  2D2V, `P ≤ 3` in 2D3V, `P ≤ 2` in 3D3V. Memory is the limit in
  practice: an SSPRK3 run holds three copies of `f` (two in the
  stepper), so 24 GB takes a little under a billion phase nodes.
  A 1D3V mesh of 64 × 24³ elements at `P = 2` is 72 million nodes,
  0.57 GB per copy; on an RTX 3090 its right-hand side takes 18 ms, the
  moment pass 2.0 ms and a full SSPRK3 step 67 ms.

### Walls

`BC_VLASOV_REFLECT` is a specular wall: a particle reaching it leaves
with its normal velocity reversed. The ghost state at velocity
$\mathbf{v}$ is the interior state at the mirrored velocity
($v_n \to -v_n$, tangential components unchanged), read at the same
wall node. That requires the velocity mesh to be symmetric about zero
along the wall-normal axis, so the mirror of a node is a node; the mesh
constructor raises otherwise. With the upwind flux the mass flux at
$v_n$ and at $-v_n$ cancel node by node, so the wall conserves mass to
round-off.

```mojo
var mesh = PhaseMesh[1, 3, P_X, P_V](
    ctx,
    lo=[X_MIN, -V_MAX, -V_MAX, -V_MAX],
    hi=[X_MAX, V_MAX, V_MAX, V_MAX],
    n=[NX, NV, NV, NV],
    periodic=[False],
    bc=[BC_VLASOV_REFLECT, BC_VLASOV_REFLECT],   # left, right
)
var x_mesh = BlockMesh[1, P_X](
    ctx, X_MIN, X_MAX, NX, periodic=False,
    bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,   # the field's wall
)
```

The wall acts on `f` only. Give the field solver its own boundary
condition on the companion `BlockMesh` (`BC_PEC_WALL` for a conductor).

The kernel relies on the advection speed along an axis being independent
of that axis's own coordinate ($v_a$ does not depend on $x_a$, $a_d$ does
not depend on $v_d$), which holds on a Cartesian phase mesh. The volume
term is then $c\,\sum_m D_{im} f_m$ and only $f$ is cached in shared
memory. Curvilinear configuration coordinates (a cylindrical radius, for
a Z-pinch) add geometric terms and are not implemented.

The rest of this page describes the hand-written 1D1V / 1D2V types.

## Module

```mojo
from warpxm.apps.vlasovmaxwell import VlasovMaxwell, VLASOV_NC   # VLASOV_NC = 1

var app = VlasovMaxwell(
    charge_mass_ratio=Real(-1.0),   # q/m, negative for electrons
    skin_depth_norm=Real(1.0),      # λ, divides the Lorentz coupling
    cfl_max=Real(0.5),
)
```

The parameter names follow upstream WARPXM. The acceleration factor is
`charge_mass_ratio / skin_depth_norm`. The current source takes $\alpha$
and $\varepsilon_0$ as separate arguments, so the driver chooses the
normalisation; the regression tests use $q/m = -1$, $\lambda = 1$,
$\varepsilon_0 = 1$, which puts the plasma frequency at 1 for unit
density.

`app.dt_max_1d1v(vx, av, dx, dv)` returns
`cfl_max / (|vx|/dx + |av|/dv)` for one node. The 1D1V / 1D2V solvers
have no `suggest_dt` (their tests use a fixed step); the generic
`DGVlasovSolver` does.

`from warpxm.physics.vlasov import *` brings in everything on this page
(see [prelude and physics](../api/prelude.md)).

## Mesh and state

```mojo
var phase_mesh = PhaseMesh1D1V[P_x, P_v](
    ctx, x_min, x_max, v_min, v_max, num_x=NX, num_v=NV, periodic_x=True,
)
var f = PhaseState1D1V[P_x, P_v, VLASOV_NC](ctx, NX, NV)

def maxwellian(x: Real, v: Real) -> Real:
    return exp(-v * v / (Real(2.0) * VT * VT)) / sqrt(TWO_PI * VT * VT)
phase_mesh.fill(f, maxwellian)
```

* Polynomial order is independent per axis (`P_x`, `P_v`, and `P_v1`,
  `P_v2` in 1D2V).
* The x axis may be periodic. The velocity axes never are: their
  boundary faces carry `BC_VLASOV_OUTFLOW`, which sets the ghost state to
  zero so nothing flows in across the velocity truncation. Choose
  $v_\text{max}$ so that $f$ is negligible there.
* The node ordering and buffer layout are in
  [`variable`](../api/variable.md#phase-space-states);
  `phase_mesh.fill` and `phase_mesh.visit_nodes` hide them
  ([`mesh`](../api/mesh.md#phase-space-meshes)).

## Spatial solver

```mojo
var spatial = DGVlasovSolver1D1V[P_x, P_v](app)
spatial.compute_rhs(ctx, phase_mesh, f, q_maxwell, rhs)
```

The fourth argument is the field state the acceleration is read from: a
`DGState[1, P_x, 6]` in Maxwell layout $(E_x, E_y, E_z, B_x, B_y, B_z)$
on the same x elements and order as the phase mesh. A Maxwell solver's
state can be passed directly, or a prescribed field can be filled in.
`DGVlasovSolver1D2V[P_x, P_v1, P_v2]` has the same call shape and reads
$E_x$, $E_y$, $B_z$.

## Moments

```mojo
var moments = VlasovMoments1D1V[P_x, P_v]()
var rho = DGState[1, P_x, 1](ctx, NX)
moments.zeroth_moment(ctx, phase_mesh, f, rho)    # ∫ f dv
moments.first_moment(ctx, phase_mesh, f, j)       # ∫ v f dv
moments.second_moment(ctx, phase_mesh, f, e_kin)  # ½ ∫ v² f dv
```

Each writes a one-component field on the x nodes. From them,
$u = J/\rho$ and $T = 2E_\text{kin}/\rho - u^2$.
`VlasovMoments1D2V` has `zeroth_moment`, `first_moment_v1`,
`first_moment_v2` and `second_moment`
($\tfrac12\int(v_1^2 + v_2^2) f$).

## Sources

* **Current into Maxwell**:
  `apply_vlasov_current_source_1d(ctx, j, rhs_maxwell, alpha, eps0)` adds
  $-\alpha J/\varepsilon_0$ to the $E_x$ RHS.
  `apply_vlasov_current_source_1d2v(ctx, j_x, j_y, rhs_maxwell, alpha, eps0)`
  does the same for $E_x$ and $E_y$.
* **BGK collisions** (1D1V):
  `VlasovBGKSource1D1V[P_x, P_v](ctx, NX)`, then
  `bgk.apply(ctx, phase_mesh, f, rhs, tau)` adds $-(f - f_M)/\tau$,
  where $f_M$ is the Maxwellian with the density, velocity and
  temperature of $f$ at each x node. The struct owns the three moment
  buffers and recomputes them on every call. A reconstructed
  temperature at or below zero is floored at $10^{-12}$.
* **Fluid to kinetic initialisation** (1D1V):
  `apply_5moment_to_kinetic_1d1v(ctx, phase_mesh, fluid, f, mass, gamma)`
  overwrites `f` with the Maxwellian matching an Euler state
  `DGState[1, P_x, EULER_NC]` at each x node
  ($n = \rho/m$, $T = P/n$, $v_\text{th} = \sqrt{T/m}$).

## Time stepping

`SSPRK3_1D1V` / `SSPRK3_1D2V` advance $f$ alone in a prescribed field.
The coupled steppers advance $(f, \mathbf{E}, \mathbf{B})$ with one
joint-RHS closure, in which the driver composes the pieces:

```mojo
comptime Phase = PhaseState1D1V[P_X, P_V, VLASOV_NC]
comptime Field = DGState[1, P_X, MAXWELL_NC]

var stepper = SSPRK3_VlasovMaxwell_1D1V[P_X, P_V](ctx, NX, NV)
var j = DGState[1, P_X, 1](ctx, NX)

def compute_joint_rhs(
    mut f_in: Phase, mut qm_in: Field, mut f_out: Phase, mut qm_out: Field,
) raises {mut j, imm}:
    spatial_vlasov.compute_rhs(ctx, phase_mesh, f_in, qm_in, f_out)
    spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
    moments.first_moment(ctx, phase_mesh, f_in, j)
    apply_vlasov_current_source_1d(ctx, j, qm_out, CHARGE_MASS_RATIO, EPS0)

stepper.step(ctx, f, q_maxwell, dt, compute_joint_rhs)
t = stepper.run_until(ctx, f, q_maxwell, t, t_end, dt, compute_joint_rhs)
```

A BGK term is one more line in the closure. The 1D2V stepper,
`SSPRK3_VlasovMaxwell_1D2V[P_x, P_v1, P_v2](ctx, NX, NV1, NV2)`, has the
same shape with two first moments and the 1D2V current source.

## Tests

| Test | What it establishes |
| ---- | ------------------- |
| `test_vlasov_scaffolding_1d1v`, `_1d2v` | mesh connectivity, state layout, app flux helpers |
| `test_vlasov_convergence_1d1v`, `_1d2v` | order $P+1$ on free streaming |
| `test_vlasov_constant_field_1d1v` | velocity-space advection in a constant $E_x$ |
| `test_vlasov_outflow_bc_1d1v` | no inflow across the velocity truncation |
| `test_vlasov_zeroth_moment_1d1v`, `_first_`, `_second_`, `test_vlasov_moments_1d2v` | moments against closed forms |
| `test_vlasov_current_source_1d` | current source touches only $E_x$, with the right coefficient |
| `test_vlasov_bgk_maxwellian_1d1v`, `test_vlasov_bgk_relaxation_1d1v` | Maxwellian is a fixed point; relaxation conserves the moments |
| `test_vlasov_5moment_to_kinetic_1d1v`, `test_vlasov_maxwell_hybrid_init_1d1v` | fluid-to-kinetic initialisation, alone and in a coupled run |
| `test_vlasov_maxwell_uniform_oscillator_1d1v` | plasma oscillation: $E$ and $J$ against $E_0\cos t$, $-E_0\sin t$ |
| `test_vlasov_maxwell_langmuir_1d1v`, `test_vlasov_maxwell_bohm_gross_1d1v` | Langmuir wave: Landau damping and the Bohm-Gross frequency |
| `test_vlasov_maxwell_weibel_1d2v`, `test_vlasov_maxwell_weibel_growth_1d2v` | 1D2V coupling with $\mathbf{v}\times\mathbf{B}$; Weibel growth rate against kinetic theory |
| `test_vlasov_generic_rhs_1d` | generic RHS equals the hand-written 1D1V and 1D2V RHS to round-off |
| `test_vlasov_rhs_consistency_1d3v` | 1D3V RHS converges to $-v_1\partial_x f - \mathbf{a}\cdot\nabla_v f$ with all six field components varying |
| `test_vlasov_convergence_1d3v` | order $P+1$ on free streaming, and on gyration about an oblique $\mathbf{B}$ with $\mathbf{E}\times\mathbf{B}$ drift and parallel acceleration |
| `test_vlasov_moments_1d3v` | the five moments of a drifting tri-Maxwellian |
| `test_vlasov_maxwell_em_wave_1d3v` | coupled 1D3V: both transverse polarisations follow the kinetic dispersion relation |
| `test_vlasov_reflecting_wall_1d3v`, `_2d2v` | specular walls: exact image solution at order $P+1$, mass conserved to round-off; walls on the second axis mirror $v_2$ only |
| `test_vlasov_reflecting_wall_fields_1d3v` | specular walls with fields: a confined equilibrium with $E_x$ pushing into both walls and an arbitrary $\mathbf{B}$; bouncing plus gyration about a wall-normal $\mathbf{B}$ |
| `test_vlasov_maxwell_walls_1d3v` | coupled run in a closed box (specular walls on $f$, PEC on the fields): mass and total energy conserved |
| `test_vlasov_rhs_consistency_2d2v` | two configuration axes: RHS consistency and moments on a `BlockMesh[2, P]` |
| `test_vlasov_maxwell_langmuir_2d2v` | coupled 2D2V: an oblique electrostatic mode follows linear theory, stays longitudinal and conserves energy |
| `test_vlasov_rhs_exact_3d3v` | 2D3V and 3D3V RHS exact to round-off on polynomial data, at $P = 1, 2$ |

Measured values are on the [Validation](../validation.md) page.

## Limits

* One species per distribution function; a second species is a second
  state and a second set of solver calls in the joint closure, with no
  stepper for that shipped.
* Cartesian phase space only: no cylindrical or other curvilinear
  configuration coordinates.
* The specular wall needs a velocity mesh symmetric about zero on the
  wall-normal axis. There is no diffuse (thermal) wall and no symmetry
  axis for curvilinear coordinates.
* 2D3V and 3D3V have a right-hand-side test only: no time-dependent or
  coupled run at those dimensions is in the suite.
* BGK and the fluid initialiser exist in 1D1V only, on the hand-written
  types.
* No slope limiter or positivity preservation for $f$.
* `profiling/prof_vlasov_1d2v.mojo` benchmarks the 1D2V step; the 1D1V
  kernels have not been profiled or tuned.
