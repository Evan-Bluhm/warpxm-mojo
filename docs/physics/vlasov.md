# Vlasov-Maxwell (kinetic)

Continuum-kinetic solver: the distribution function $f(x, \mathbf{v}, t)$
of one species is evolved on a phase-space DG mesh and coupled to
Maxwell's equations through its current. Implemented in 1D1V
$(x, v_x)$ and 1D2V $(x, v_x, v_y)$.

This is a separate code path from the fluid solvers: it has its own
meshes, state types, kernels and steppers. The Maxwell field lives on an
ordinary `BlockMesh[1, P_x]` / `DGState[1, P_x, MAXWELL_NC]`.

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

The flux is linear in $f$, so every face uses the upwind (Rusanov) flux
with the local coordinate or acceleration as the wave speed.

Maxwell receives the species current:

$$
\partial_t \mathbf{E} \mathrel{+}= -\frac{\alpha}{\varepsilon_0}\,\mathbf{J},
\qquad \mathbf{J}(x) = \int \mathbf{v}\, f\, d\mathbf{v},
$$

with $\alpha = q/m$. In 1D1V only $E_x$ is driven; in 1D2V both $E_x$
and $E_y$.

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
`cfl_max / (|vx|/dx + |av|/dv)` for one node. There is no
`suggest_dt` for the kinetic solvers: the tests use a fixed step.

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

Measured values are on the [Validation](../validation.md) page.

## Limits

* One species per distribution function; a second species is a second
  state and a second set of solver calls in the joint closure, with no
  stepper for that shipped.
* No 2D or 3D configuration space, and no third velocity dimension.
* BGK and the fluid initialiser exist in 1D1V only.
* No slope limiter or positivity preservation for $f$.
* `profiling/prof_vlasov_1d2v.mojo` benchmarks the 1D2V step; the 1D1V
  kernels have not been profiled or tuned.
