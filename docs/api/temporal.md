# `warpxm.temporal`

Explicit time integrators. The architecture mirrors upstream WARPXM's
`spatial_solver + variable_adjuster` separation: the stepper knows no
physics. The spatial RHS and the adjusters (slope limiter, positivity
preserver) are composed at the call site as closures (nested `def`s with
a capture list).

## Files

| File | Stepper | Use |
| ---- | ------- | --- |
| `ssprk3.mojo` | `SSPRK3[NDIM, P, NC]`, `no_op_adjust[NDIM, P, NC]` | any single-state problem in 1D, 2D or 3D |
| `dormand_prince.mojo` | `DormandPrince45[NDIM, P, NC]` | fixed-step or adaptive 5(4) RK, any single-state problem |
| `ssprk3_twofluid.mojo` | `SSPRK3_TwoFluid[NDIM, P]` | one Euler fluid + Maxwell |
| `ssprk3_plasma.mojo` | `SSPRK3_Plasma[NDIM, P]` | two Euler species + Maxwell |
| `ssprk3_phase.mojo` | `SSPRK3Phase[XDIM, VDIM, P_x, P_v, NC]`, `no_op_adjust_phase` | a phase-space state alone, any dimensions |
| `ssprk3_vlasov_maxwell.mojo` | `SSPRK3VlasovMaxwell[XDIM, VDIM, P_x, P_v]` | Vlasov + Maxwell, any dimensions |
| `ssprk3_1d1v.mojo`, `ssprk3_1d2v.mojo` | `SSPRK3_1D1V[P_x, P_v, NC]`, `SSPRK3_1D2V[P_x, P_v1, P_v2, NC]` | a phase-space state alone |
| `ssprk3_vlasov_maxwell_1d1v.mojo`, `..._1d2v.mojo` | `SSPRK3_VlasovMaxwell_1D1V[P_x, P_v]`, `SSPRK3_VlasovMaxwell_1D2V[P_x, P_v1, P_v2]` | Vlasov + 1D Maxwell |

The state buffer is flat in every dimension, so the RK combine kernels
do not depend on `NDIM`; it only fixes the state type. Each stepper owns
its scratch states and is constructed as `Stepper[...](ctx, num_elements)`
(the kinetic ones take the phase-mesh element counts).

## `SSPRK3`

```mojo
from warpxm.temporal.ssprk3 import SSPRK3, no_op_adjust

comptime State = DGState[2, P, EULER_NC]

var spatial = DGEulerSolver[2, P](app)
var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements)
var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    # source kernels go here (gravity, GLM damping, ...)

def adjust(mut q_: State) raises {mut limiter, imm}:
    limiter.apply(ctx, mesh, q_)

stepper.step(ctx, q, dt, compute_rhs, adjust)
```

`compute_rhs_fn(q_in, q_out)` writes the time derivative of `q_in` to
`q_out`. `adjust_fn(q)` runs on the incoming state and after each stage
value is written. Both closures are alive during the call, so they must
not both capture the same object mutably: name `mut` only for what a
closure mutates (the limiter's scratch buffers) and leave the rest to
`imm`.

For smooth problems pass `no_op_adjust[NDIM, P, NC]` as the adjuster.

### `run_until`

```mojo
t = stepper.run_until(ctx, q, t, t_target, dt, compute_rhs, adjust)
```

Advances from `t` to `t_target` in steps of `dt`, shortens the last step
to land on `t_target`, and returns the time reached. A remainder smaller
than `1e-9 * dt` is treated as round-off and not stepped. Every stepper
in this package has `run_until`, with the argument list of its `step`
and `t, t_target` inserted before `dt`.

### `run_until_cfl`

```mojo
def suggest_dt(mut q_: State) raises {imm} -> Real:
    return spatial.suggest_dt(ctx, mesh, q_)

t = stepper.run_until_cfl(
    ctx, q, t, t_target, compute_rhs, adjust, suggest_dt, dt_interval=10,
)
```

Advances to `t_target` with the step returned by `suggest_dt_fn(u)`,
shortening the last step, and returns `t_target`. The suggestion costs a
GPU reduction and a host read-back, so it is refreshed only every
`dt_interval` steps (default 1). Raise the interval on smooth problems;
keep it small where the wave speed changes quickly. Only `SSPRK3` has
this method. See [`spatial`](spatial.md#cfl-time-step) for the bound.

## Source-coupled problems

Composition is several kernel calls inside `compute_rhs`:

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

The Rusanov / HLLD choice is on the spatial solver type
(`DGImhdSolver[2, P, USE_HLLD=True]`), not on the stepper.

## `DormandPrince45`

```mojo
from warpxm.temporal.dormand_prince import DormandPrince45

var stepper = DormandPrince45[2, P, EULER_NC](ctx, mesh.num_elements)

stepper.step(ctx, q, dt, compute_rhs)                    # fixed step
var ret = stepper.step_adaptive(
    ctx, q, dt_try, compute_rhs, atol=Real(1e-8), rtol=Real(1e-5),
)
var dt_used = ret[0]
var dt_next = ret[1]
```

There is no adjuster argument. `step_adaptive` retries with a smaller
step until the scaled RMS error is below 1 (at most `max_attempts`,
default 10) and returns `(dt_used, dt_next)`. Defaults: `atol=1e-6`,
`rtol=1e-3`. `run_until` uses the fixed `step`; an adaptive loop is
written by the driver:

```mojo
while t < t_final - Real(1.0e-12):
    var dt_try = dt if (t + dt) < t_final else (t_final - t)
    var ret = stepper.step_adaptive(ctx, q, dt_try, compute_rhs)
    t += ret[0]
    dt = ret[1]
```

The error norm $\sqrt{\sum (\text{err}/\text{sc})^2 / N}$ is reduced on
the GPU by `_dp45_sum_sq_kernel`; one scalar reaches the host per
attempt. See [Dormand-Prince](../numerical-methods/dormand-prince.md).

## Coupled fluid + Maxwell steppers

`SSPRK3_TwoFluid[NDIM, P]` (one fluid + Maxwell) and
`SSPRK3_Plasma[NDIM, P]` (two species + Maxwell) take one
`compute_joint_rhs_fn` closure that writes the full RHS (hyperbolic terms
plus coupling sources) of every state. Source ordering and the
$\alpha$ / $\varepsilon_0$ values live in the driver:

```mojo
from warpxm.temporal.ssprk3_plasma import SSPRK3_Plasma

comptime Fluid = DGState[2, P, EULER_NC]
comptime Field = DGState[2, P, MAXWELL_NC]

def compute_joint_rhs(
    mut q1_in: Fluid, mut q2_in: Fluid, mut qm_in: Field,
    mut q1_out: Fluid, mut q2_out: Fluid, mut qm_out: Field,
) raises {imm}:
    fluid_spatial.compute_rhs(ctx, mesh, q1_in, q1_out)
    fluid_spatial.compute_rhs(ctx, mesh, q2_in, q2_out)
    maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
    apply_lorentz_source(ctx, q1_in, qm_in, q1_out, ALPHA_1)
    apply_lorentz_source(ctx, q2_in, qm_in, q2_out, ALPHA_2)
    apply_current_source(ctx, q1_in, qm_out, ALPHA_1, EPS0)
    apply_current_source(ctx, q2_in, qm_out, ALPHA_2, EPS0)

var stepper = SSPRK3_Plasma[2, P](ctx, mesh.num_elements)
stepper.step(ctx, q_s1, q_s2, q_maxwell, dt, compute_joint_rhs)
t = stepper.run_until(ctx, q_s1, q_s2, q_maxwell, t, t_end, dt, compute_joint_rhs)
```

`SSPRK3_TwoFluid` is the two-state shape
`compute_joint_rhs(qf_in, qm_in, qf_out, qm_out)` with
`step(ctx, q_fluid, q_maxwell, dt, compute_joint_rhs)`. Neither takes an
adjuster, and neither has a CFL-driven variant: the plasma-frequency
source terms, not only the wave speeds, limit the step.

## Kinetic steppers

`SSPRK3Phase` and `SSPRK3VlasovMaxwell` are the generic kinetic steppers
(constructed from `mesh.num_phase_elements`, plus `mesh.num_x_elements`
for the coupled one). Besides `step` and `run_until` they have
`run_until_cfl`, whose `suggest_dt` closure takes the phase state (and
the Maxwell state, for the coupled stepper). They keep two scratch
copies of the state rather than four (`stage` and `rhs`), because the
number of copies of `f` is what limits a kinetic run. `CflPacer`
(`temporal/pacing.mojo`) is the step-size bookkeeping shared by the
`run_until_cfl` loops. The original per-dimension steppers follow.

`SSPRK3_1D1V` / `SSPRK3_1D2V` advance one phase-space state with the
same `step(ctx, f, dt, compute_rhs, adjust)` shape (adjuster
`no_op_adjust_1d1v` / `no_op_adjust_1d2v`). `SSPRK3_VlasovMaxwell_1D1V` /
`_1D2V` advance `(f, q_maxwell)` together with one joint-RHS closure:
`step(ctx, q_phase, q_maxwell, dt, compute_joint_rhs)`. See
[Vlasov-Maxwell](../physics/vlasov.md).
