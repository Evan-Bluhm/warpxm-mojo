# Time integration

warpxm-mojo ships two explicit Runge-Kutta integrators. The library
exposes one **generic stepper per dimension** — the spatial-RHS
evaluation, source terms, slope limiter, positivity preserver, and any
other adjusters are composed at the call site via `@parameter def`
closures. That's upstream WARPXM's `spatial_solver + variable_adjuster`
pattern, expressed in Mojo's compile-time function-parameter system.

## SSPRK3 — the workhorse

[Three-stage Shu-Osher SSPRK3](../numerical-methods/ssprk3.md) is the
default for almost everything. It's strong-stability-preserving (any
property the forward-Euler step preserves — positivity, TVD bounds —
SSPRK3 also preserves) and third-order accurate in time.

```mojo
from warpxm.temporal.ssprk3_2d import SSPRK3_2D

var stepper = SSPRK3_2D[P, NC](ctx, mesh.num_elements)

@parameter
def compute_rhs(mut q_in, mut q_out) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

@parameter
def adjust(mut q_) raises:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)

stepper.step[compute_rhs, adjust](ctx, q, dt)
```

Three steppers cover every dimension:

| Stepper | Used by |
| ------- | ------- |
| `SSPRK3_1D[P, NC]` | Every 1D problem (advection, Euler) |
| `SSPRK3_2D[P, NC]` | Every 2D problem (Euler, MHD, Maxwell, GLM-MHD) |
| `SSPRK3_3D[P, NC]` | Every 3D problem |

The plasma family (`SSPRK3_Plasma_{2,3}D`, `SSPRK3_Twofluid_{2,3}D`)
stays specialised because it juggles three coupled states (two fluid
species plus Maxwell) — see [Plasma](../physics/plasma.md).

## Dormand-Prince RK45 — adaptive

[Five-stage Dormand-Prince 5(4)](../numerical-methods/dormand-prince.md)
adds an embedded fourth-order error estimate so the step size can shrink
near steep gradients and relax in smooth regions.

```mojo
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
var stepper = DormandPrince45_1D[P, NC](ctx, mesh.num_elements)

# Fixed-step:
stepper.step(ctx, mesh, spatial, q, dt)

# Adaptive:
var ret = stepper.step_adaptive(ctx, mesh, spatial, q, dt_try,
                                atol=1e-8, rtol=1e-6)
var dt_used = ret[0]
var dt_next = ret[1]
```

On the smooth-advection benchmark adaptive DP45 converges in **~10×
fewer steps than fixed-Δt SSPRK3** at the same error tolerance.

DP45 currently uses the pre-refactor direct-spatial-solver API; the
2D Euler variant lives in `dormand_prince_euler_2d.mojo`. Other physics
× dimension combinations and migration to the same closure pattern as
SSPRK3 are open follow-ups.

## Composing sources and adjusters

The two-closure pattern absorbs everything that was previously a
specialised stepper variant.

### Sources (gravity, Lorentz, GLM damping, …)

Fold them into `compute_rhs` after the main spatial RHS call:

```mojo
@parameter
def compute_rhs(mut q_in, mut q_out) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    apply_gravity_source_2d[P](ctx, q_in, q_out, gx, gy)
    apply_glm_damping_2d[P](ctx, q_in, q_out, c_h, c_p_sq)
```

Sources add to `q_out` rather than overwriting, so the spatial RHS and
each source contribute additively — same as the multi-app composition
in upstream.

### Limiters and positivity preservers

Apply between RK stages via `adjust`:

```mojo
@parameter
def adjust(mut q_) raises:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)
```

Order matters: the limiter is shape-preserving on conserved variables;
the positivity preserver enforces ρ > 0, P > 0 by checking primitives.
The stepper calls `adjust` once on the input plus once after each of
the three RK stages — every spatial RHS evaluation sees an adjusted
state. See [Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md)
and [Zhang-Shu positivity preserver](../numerical-methods/positivity.md).

### No adjuster

Smooth problems (advection, Maxwell, smooth Euler) supply a no-op:

```mojo
@parameter
def no_adjust(mut q_) raises:
    pass
```

## Choosing a stepper

```text
Any single-state 1D / 2D / 3D problem (advection, Euler, Maxwell, MHD, GLM-MHD)
    → SSPRK3_<D>D[P, NC]
    → compose limiter / positivity / sources in the closures

Smooth problem, want adaptive Δt
    → DormandPrince45_1D[P, NC]  (1D)
    → DormandPrince45_Euler_2D[P]  (2D Euler)

Plasma (two species)
    → SSPRK3_Plasma_{2,3}D[P]
```
