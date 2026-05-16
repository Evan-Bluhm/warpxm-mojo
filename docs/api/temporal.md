# `warpxm.temporal`

Explicit time integrators. The architecture mirrors upstream WARPXM's
`spatial_solver + variable_adjuster` separation: one generic stepper
per dimension; the spatial RHS evaluation and the adjuster list (slope
limiter, positivity preserver, GLM damping, source terms, …) are
composed at the call site via `@parameter def` closures.

## Files

| File | Stepper(s) | Use |
| ---- | ---------- | --- |
| `ssprk3.mojo` | `_euler_step_kernel`, `_ssprk3_combine_kernel` | Shared dimension-agnostic combine kernels imported by every SSPRK3 variant. No struct. |
| `ssprk3_1d.mojo` | `SSPRK3_1D[P, NC]` | Every 1D problem (advection, Euler, …) |
| `ssprk3_2d.mojo` | `SSPRK3_2D[P, NC]` | Every 2D problem (Euler, Maxwell, MHD, GLM-MHD, …) |
| `ssprk3_3d.mojo` | `SSPRK3_3D[P, NC]` | Every 3D problem |
| `dormand_prince_1d.mojo` | `DormandPrince45_1D[P, NC]` + dimension-agnostic combine helpers `_combine1`–`_combine5`, the per-cell error kernel `_dp45_error_kernel`, and the single-block reduction kernel `_dp45_sum_sq_kernel` — all three reused by 2D/3D | Adaptive 5(4) RK in 1D |
| `dormand_prince_2d.mojo` | `DormandPrince45_2D[P, NC]` | Adaptive 5(4) RK in 2D — any physics |
| `dormand_prince_3d.mojo` | `DormandPrince45_3D[P, NC]` | Adaptive 5(4) RK in 3D — any physics |
| `ssprk3_plasma_{2,3}d.mojo` | `SSPRK3_Plasma_{2,3}D[P]` | Two-species plasma — three coupled states, takes a single `compute_joint_rhs_fn` closure |
| `ssprk3_twofluid_{2,3}d.mojo` | `SSPRK3_TwoFluid_{2,3}D[P]` | Single-fluid plasma (one fluid + Maxwell), takes a `compute_joint_rhs_fn` closure |

## The two-closure pattern

`SSPRK3_{1,2,3}D[P, NC].step` takes two comptime closure parameters:

```mojo
def step[
    compute_rhs_fn: def(mut DGStateNd[P, NC], mut DGStateNd[P, NC])
                      raises capturing[_] -> None,
    adjust_fn:      def(mut DGStateNd[P, NC])
                      raises capturing[_] -> None,
](mut self, var ctx, mut u, dt) raises
```

`compute_rhs_fn(q_in, q_out)` evaluates the spatial RHS for the supplied
input and writes it to `q_out`. `adjust_fn(q)` runs after each RK stage
value is written. Both closures capture whatever the driver needs
(`spatial`, `mesh`, `limiter`, `positivity`, source-kernel parameters,
etc.).

This is the upstream WARPXM `spatial_solver + variable_adjuster`
pattern expressed in Mojo: the library has no per-physics knowledge,
the driver composes physics, sources, limiters, and positivity steps
in a few lines at the call site.

## Driver pattern (every problem looks like this)

```mojo
var spatial = DG<Physics>Solver<D>D[P](app)
var stepper = SSPRK3_<D>D[P, <Physics>_NC](ctx, mesh.num_elements)

@parameter
def compute_rhs(
    mut q_in: DGState<D>D[P, <Physics>_NC],
    mut q_out: DGState<D>D[P, <Physics>_NC],
) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    # ... add source kernels here (gravity, GLM damping, Lorentz, …)

@parameter
def adjust(mut q_: DGState<D>D[P, <Physics>_NC]) raises:
    limiter.apply(ctx, mesh, q_)        # zero, one, or many
    positivity.apply(ctx, mesh, q_)

# Time loop
while t < t_final:
    stepper.step[compute_rhs, adjust](ctx, q, dt)
    t += dt
```

For smooth problems (advection, Maxwell, smooth Euler), supply a no-op
adjuster:

```mojo
@parameter
def no_adjust(mut q_: DGState<D>D[P, <Physics>_NC]) raises:
    pass
```

## Source-coupled problems

The "multi-app" composition from upstream is just multiple kernel calls
inside `compute_rhs`:

```mojo
# Euler with gravity:
@parameter
def compute_rhs(mut q_in, mut q_out) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    apply_gravity_source_2d[P](ctx, q_in, q_out, gx, gy)

# Ideal MHD with GLM cleaning:
@parameter
def compute_rhs(mut q_in, mut q_out) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    apply_glm_damping_2d[P](ctx, q_in, q_out, c_h, c_p_sq)
```

## MHD with HLLD vs Rusanov

`HLLD` selection is on the spatial-solver type, not the stepper:

```mojo
var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)
# ... compute_rhs / adjust closures ...
stepper.step[compute_rhs, adjust](ctx, q, dt)
```

The closure picks up `spatial`, so whether you used `USE_HLLD=True` or
the default is captured automatically.

## Adaptive RK45 (Dormand-Prince)

`DormandPrince45_{1,2,3}D[P, NC]` use the same two-closure pattern as
SSPRK3 — `step[compute_rhs_fn]` for fixed-dt and
`step_adaptive[compute_rhs_fn]` for embedded 4th-order error control.
The combine kernels (`_combine1`–`_combine5`) and the per-cell error
kernel live in `dormand_prince_1d.mojo` and are shared across
dimensions because they operate on flat buffers. The RMS scaled
error norm `√(Σ(err/sc)²/N)` is reduced fully on the GPU by a single
block tree-reduction (`_dp45_sum_sq_kernel`); only the final scalar
`Σ(err/sc)²` lands on the host per attempt — no `map_to_host` of the
full state buffers. Validated bit-comparable (modulo floating-point
sum-order on one test) across all 5 physics × all dimensions; see
[Dormand-Prince](../numerical-methods/dormand-prince.md).

## Plasma steppers

`SSPRK3_Plasma_{2,3}D[P]` and `SSPRK3_TwoFluid_{2,3}D[P]` extend the
closure pattern to multiple coupled states (two fluid species + Maxwell,
or one fluid + Maxwell). The driver supplies a single
`compute_joint_rhs_fn(*states_in, *rhs_out)` closure that writes the
combined hyperbolic + source RHS for all states. Source ordering,
α / ε₀ values, and the choice of which sources are active all live in
the driver:

```mojo
@parameter
def compute_joint_rhs(
    mut q1_in: DGState2D[P, EULER_NC],
    mut q2_in: DGState2D[P, EULER_NC],
    mut qm_in: DGState2D[P, MAXWELL_NC],
    mut q1_out: DGState2D[P, EULER_NC],
    mut q2_out: DGState2D[P, EULER_NC],
    mut qm_out: DGState2D[P, MAXWELL_NC],
) raises:
    fluid_spatial.compute_rhs(ctx, mesh, q1_in, q1_out)
    fluid_spatial.compute_rhs(ctx, mesh, q2_in, q2_out)
    maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
    apply_lorentz_source_2d[P](ctx, q1_in, qm_in, q1_out, ALPHA_1)
    apply_lorentz_source_2d[P](ctx, q2_in, qm_in, q2_out, ALPHA_2)
    apply_current_source_2d[P](ctx, q1_in, qm_out, ALPHA_1, EPS0)
    apply_current_source_2d[P](ctx, q2_in, qm_out, ALPHA_2, EPS0)

stepper.step[compute_joint_rhs](ctx, q_s1, q_s2, q_maxwell, dt)
```

`SSPRK3_TwoFluid_{2,3}D` is the 2-state shape `compute_joint_rhs(qf_in,
qm_in, qf_out, qm_out)`; same pattern, fewer arguments.
