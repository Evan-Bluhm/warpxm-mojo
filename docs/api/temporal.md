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
| `dormand_prince_1d.mojo` | `DormandPrince45_1D[P, NC]` | Adaptive 5(4) RK in 1D |
| `dormand_prince_euler_2d.mojo` | `DormandPrince45_Euler_2D[P]` | Adaptive 5(4) RK for 2D Euler |
| `ssprk3_plasma_{2,3}d.mojo` | `SSPRK3_Plasma_{2,3}D[P]` | Two-species plasma — three coupled states, kept specialized |
| `ssprk3_twofluid_{2,3}d.mojo` | `SSPRK3_Twofluid_{2,3}D[P]` | Single-fluid plasma (one fluid + Maxwell) |

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

`DormandPrince45_1D[P, NC]` and `DormandPrince45_Euler_2D[P]` keep their
direct API for now (they pre-date the closure pattern and have an
embedded fixed-shape error estimator). Migration to the same
closure-based API is a follow-up. See
[Dormand-Prince](../numerical-methods/dormand-prince.md).

## Plasma steppers

`SSPRK3_Plasma_{2,3}D[P]` and `SSPRK3_Twofluid_{2,3}D[P]` are kept
specialised because they juggle three coupled states (two fluid species
plus Maxwell) and a fixed-shape Lorentz / current-source coupling. They
don't fit the single-`u`-buffer signature of `SSPRK3_{1,2,3}D`.
