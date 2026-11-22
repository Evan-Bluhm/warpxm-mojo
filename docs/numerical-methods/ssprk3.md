# SSPRK3 time stepper

Three-stage strong-stability-preserving Runge-Kutta in the Shu-Osher
form. The default integrator in warpxm-mojo.

Citation: Gottlieb, S., Shu, C.-W. & Tadmor, E., "Strong stability-
preserving high-order time discretization methods", *SIAM Review* 43
(2001) 89–112.

## The stages

For a semi-discrete ODE $\dot{\mathbf{u}} = L(\mathbf{u})$:

$$
\mathbf{u}^{(1)} = \mathbf{u}^{n} + \Delta t\, L(\mathbf{u}^{n}),
$$

$$
\mathbf{u}^{(2)} = \tfrac{3}{4}\mathbf{u}^{n} + \tfrac{1}{4}\!\left(\mathbf{u}^{(1)} + \Delta t\, L(\mathbf{u}^{(1)})\right),
$$

$$
\mathbf{u}^{n+1} = \tfrac{1}{3}\mathbf{u}^{n} + \tfrac{2}{3}\!\left(\mathbf{u}^{(2)} + \Delta t\, L(\mathbf{u}^{(2)})\right).
$$

Each stage is a *convex combination* of a forward-Euler step from a
previous stage with the original $\mathbf{u}^{n}$. The "SSP" property
says: if forward-Euler preserves a convex bound or monotonicity
property (positivity, TVD, ENO), the three-stage RK update preserves
the same property under the same CFL.

## Properties

* **Order 3** in time on smooth problems.
* **Strong-stability-preserving** at $\text{CFL} \leq 1$ relative to
  forward Euler.
* **Three RHS evaluations** per step (one per stage).
* **GPU-friendly**: each stage is a kernel launch + an `enqueue_copy`
  for the snapshot. Two shared kernels live in
  `temporal/ssprk3.mojo` and are reused by every SSPRK3 variant
  (including the plasma, two-fluid and kinetic steppers):
    * `_euler_step_kernel` — `output[i] = x[i] + dt · rhs[i]` (used
      for stage 1, where aliasing prevents the more general combine).
    * `_ssprk3_combine_kernel` — `output[i] = a · u₀[i] + b · (u_stage[i]
      + dt · rhs[i])` (used for stages 2 and 3 with $(a, b) = (3/4,
      1/4)$ then $(1/3, 2/3)$).

## Steppers in `temporal/`

`temporal/ssprk3.mojo` holds the one generic stepper,
`SSPRK3[NDIM, P, NC]`, for every single-state problem in 1D, 2D and 3D.
The state buffer is flat, so nothing in the RK combine depends on the
dimension. Besides `step` it has `run_until` (fixed step to a target
time) and `run_until_cfl` (step from `suggest_dt`).

Limiters, positivity preservers, source kernels (gravity, GLM damping,
…) are composed at the call site as closures (nested `def`s with a capture list) — see
[Time integration](../concepts/time-integration.md). The plasma family
(`ssprk3_plasma.mojo`, `ssprk3_twofluid.mojo`, also generic over the
dimension) is kept separate because it advances three (or two) coupled
states.

## CFL constraint

For a DG scheme of polynomial order $P$ on a uniform mesh the stable
step is

$$
\Delta t \le \frac{C_{\text{CFL}}}{(2P + 1)\,\sum_d \lambda_d / \Delta x_d},
$$

where $\lambda_d$ bounds the signal speed along axis $d$. This is what
`solver.suggest_dt(...)` returns, with $C_{\text{CFL}}$ the app's
`cfl_max`.

Measured with this code's GLL-collocation DG and SSPRK3 on linear
advection (`tests/test_cfl_timestep.mojo`): stable up to
$C_{\text{CFL}} \approx 1.5$ for $P = 1..4$, unstable by 2 for
$P \ge 3$. The test pins the two ends: energy is non-increasing at
$C_{\text{CFL}} = 1$ and the run blows up at 4, for every $P$.

In practice:

* 0.5 for problems with shocks. The limiter and positivity preserver
  assume a forward-Euler-stable step, and the nonlinear solvers use one
  direction-independent speed bound.
* up to 1 for smooth linear problems.

For $P = 2$, $\Delta x = 1/64$, $\lambda = 1$ in 2D and
$C_{\text{CFL}} = 0.5$ that gives $\Delta t \approx 7.8\times10^{-4}$.

## When you'd want Dormand-Prince instead

* You're running a smooth problem with strongly varying time scales,
  where most of the run can take large steps and a few regions need
  small ones. SSPRK3 with fixed $\Delta t$ pessimises for the
  smallest required step.
* You want an automatic error tolerance instead of hand-tuning
  $\Delta t$.

In both cases see [Dormand-Prince](dormand-prince.md).
