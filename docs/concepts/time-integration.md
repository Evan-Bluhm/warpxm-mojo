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
| `SSPRK3_1D[P, NC]` | Every 1D problem (advection, Euler, Maxwell) |
| `SSPRK3_2D[P, NC]` | Every 2D problem (Euler, MHD, Maxwell, GLM-MHD) |
| `SSPRK3_3D[P, NC]` | Every 3D problem |

The plasma family (`SSPRK3_Plasma_{1,2,3}D`, `SSPRK3_TwoFluid_{1,2,3}D`)
extends the same closure pattern to multiple coupled states. Each takes
a single `compute_joint_rhs_fn(*states_in, *rhs_out)` closure that
writes the combined hyperbolic + source RHS for the (fluid₁, fluid₂,
Maxwell) or (fluid, Maxwell) tuple. See [Plasma](../physics/plasma.md).

## Dormand-Prince RK45 — adaptive

[Seven-stage Dormand-Prince 5(4)](../numerical-methods/dormand-prince.md)
delivers a fifth-order solution alongside an embedded fourth-order
estimate, so the step size can shrink near steep gradients and relax
in smooth regions.

```mojo
from warpxm.temporal.dormand_prince_2d import DormandPrince45_2D
var stepper = DormandPrince45_2D[P, NC](ctx, mesh.num_elements)

@parameter
def compute_rhs(mut q_in, mut q_out) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

# Fixed-step:
stepper.step[compute_rhs](ctx, q, dt)

# Adaptive:
var ret = stepper.step_adaptive[compute_rhs](
    ctx, q, dt_try, atol=1e-8, rtol=1e-6,
)
var dt_used = ret[0]
var dt_next = ret[1]
```

DP45 uses the same two-closure pattern as SSPRK3 and is available in
all three dimensions (`DormandPrince45_{1,2,3}D[P, NC]`). On smooth
problems the controller takes **6-40× fewer steps** than fixed-dt
SSPRK3 at comparable error tolerance — biggest wins on linear systems
like Maxwell (40× in 3D).

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
    → DormandPrince45_<D>D[P, NC]  (any dim, any physics)
    → same two-closure API as SSPRK3

Plasma (two species)        → SSPRK3_Plasma_{1,2,3}D[P]
Single fluid + Maxwell      → SSPRK3_TwoFluid_{1,2,3}D[P]
    → both take compute_joint_rhs_fn(*states_in, *rhs_out)
```
