# Time integration

warpxm-mojo ships two explicit Runge-Kutta integrators. Each is **one
generic stepper**, parameterised on the spatial dimension — the
spatial-RHS evaluation, source terms, slope limiter, positivity
preserver, and any other adjusters are composed at the call site via
nested `def` closures. That's upstream WARPXM's `spatial_solver + variable_adjuster`
pattern, expressed in Mojo's compile-time function-parameter system.

## SSPRK3 — the workhorse

[Three-stage Shu-Osher SSPRK3](../numerical-methods/ssprk3.md) is the
default for almost everything. It's strong-stability-preserving (any
property the forward-Euler step preserves — positivity, TVD bounds —
SSPRK3 also preserves) and third-order accurate in time.

```mojo
from warpxm.temporal.ssprk3 import SSPRK3

comptime State = DGState[2, P, NC]
var stepper = SSPRK3[2, P, NC](ctx, mesh.num_elements)

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)

stepper.step(ctx, q, dt, compute_rhs, adjust)
```

`SSPRK3[NDIM, P, NC]` covers every single-state problem in 1D, 2D and
3D: the state buffer is flat, so the RK combine kernels do not depend on
the dimension.

## Choosing the time step

A run rarely calls `step` directly. Two drivers advance to a target
time:

```mojo
# Fixed step, last step shortened to land on t_target:
t = stepper.run_until(ctx, q, t, t_target, dt, compute_rhs, adjust)

# CFL-driven:
def suggest_dt(mut q_: State) raises {imm} -> Real:
    return spatial.suggest_dt(ctx, mesh, q_)
t = stepper.run_until_cfl(
    ctx, q, t, t_target, compute_rhs, adjust, suggest_dt, dt_interval=10,
)
```

`suggest_dt` returns

$$
\Delta t = \frac{\text{cfl\_max}}{(2P + 1)\,\sum_d \lambda_d/\Delta x_d},
$$

with $\lambda$ the largest signal speed in the current state (reduced on
the GPU) and `cfl_max` taken from the app. With SSPRK3, linear advection
is stable up to `cfl_max` of about 1.5 for $P \le 4$ and unstable by 2
for $P \ge 3$; use 0.5 for problems with shocks. Because the suggestion
needs a host read-back, `dt_interval` controls how often it is
refreshed. Examples use the CFL-driven form; tests that pin a step count
use a fixed `dt`. Details: [`spatial`](../api/spatial.md#cfl-time-step)
and [`temporal`](../api/temporal.md).

The plasma family (`SSPRK3_Plasma[NDIM, P]`, `SSPRK3_TwoFluid[NDIM, P]`)
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
from warpxm.temporal.dormand_prince import DormandPrince45
var stepper = DormandPrince45[2, P, NC](ctx, mesh.num_elements)

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

# Fixed-step:
stepper.step(ctx, q, dt, compute_rhs)

# Adaptive:
var ret = stepper.step_adaptive(
    ctx, q, dt_try, compute_rhs, atol=Real(1e-8), rtol=Real(1e-6),
)
var dt_used = ret[0]
var dt_next = ret[1]
```

DP45 takes the same `compute_rhs` closure as SSPRK3 (it has no adjuster
argument) and is one struct for all three dimensions
(`DormandPrince45[NDIM, P, NC]`). On smooth
problems the controller takes **6-40× fewer steps** than fixed-dt
SSPRK3 at comparable error tolerance — biggest wins on linear systems
like Maxwell (40× in 3D).

## Composing sources and adjusters

The two-closure pattern absorbs everything that was previously a
specialised stepper variant.

### Sources (gravity, Lorentz, GLM damping, …)

Fold them into `compute_rhs` after the main spatial RHS call:

```mojo
# Euler with gravity:
def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    apply_gravity_source(ctx, q_in, q_out, gx, gy)

# GLM-MHD with ψ damping:
def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    apply_glm_damping(ctx, q_in, q_out, c_h, c_p_sq)
```

Sources add to `q_out` rather than overwriting, so the spatial RHS and
each source contribute additively — same as the multi-app composition
in upstream.

### Limiters and positivity preservers

Apply between RK stages via `adjust`:

```mojo
def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
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

Smooth problems (advection, Maxwell, smooth Euler) pass the library's
no-op:

```mojo
from warpxm.temporal.ssprk3 import no_op_adjust

stepper.step(ctx, q, dt, compute_rhs, no_op_adjust[2, P, NC])
```

## Choosing a stepper

```text
Any single-state 1D / 2D / 3D problem (advection, Euler, Maxwell, MHD, GLM-MHD)
    → SSPRK3[NDIM, P, NC]
    → compose limiter / positivity / sources in the closures
    → run_until_cfl for a CFL-driven step, run_until for a fixed one

Smooth problem, want error-controlled Δt
    → DormandPrince45[NDIM, P, NC]  (any dim, any physics)
    → same compute_rhs closure as SSPRK3, no adjuster

Plasma (two species)        → SSPRK3_Plasma[NDIM, P]
Single fluid + Maxwell      → SSPRK3_TwoFluid[NDIM, P]
    → both take compute_joint_rhs_fn(*states_in, *rhs_out)

Kinetic (Vlasov, Vlasov-Maxwell)
    → SSPRK3_1D1V / SSPRK3_1D2V, SSPRK3_VlasovMaxwell_1D1V / _1D2V
```
