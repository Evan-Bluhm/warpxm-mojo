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
  `temporal/ssprk3.mojo` and are reused by every `SSPRK3_*` variant
  (including the plasma / two-fluid steppers):
    * `_euler_step_kernel` — `output[i] = x[i] + dt · rhs[i]` (used
      for stage 1, where aliasing prevents the more general combine).
    * `_ssprk3_combine_kernel` — `output[i] = a · u₀[i] + b · (u_stage[i]
      + dt · rhs[i])` (used for stages 2 and 3 with $(a, b) = (3/4,
      1/4)$ then $(1/3, 2/3)$).

## Steppers in `temporal/`

One generic stepper per dimension:

| File | Stepper | Use |
| ---- | ------- | --- |
| `ssprk3_1d.mojo` | `SSPRK3_1D[P, NC]` | Every 1D problem |
| `ssprk3_2d.mojo` | `SSPRK3_2D[P, NC]` | Every 2D problem |
| `ssprk3_3d.mojo` | `SSPRK3_3D[P, NC]` | Every 3D problem |

Limiters, positivity preservers, source kernels (gravity, GLM damping,
…) are composed at the call site as closures (nested `def`s with a capture list) — see
[Time integration](../concepts/time-integration.md). The plasma family
(`ssprk3_plasma_{1,2,3}d`, `ssprk3_twofluid_{1,2,3}d`) is kept
specialised because it juggles three (or two) coupled states.

## CFL constraint

For a DG scheme with polynomial order $P$ on uniform mesh spacing
$\Delta x$, the empirical stable $\Delta t$ for SSPRK3 + Rusanov is

$$
\Delta t \lesssim \frac{C_{\text{CFL}}}{(2P + 1)} \cdot \frac{\Delta x}{\lambda_{\max}},
$$

with $C_{\text{CFL}} \approx 1/3$ working for everything in the repo.
For $P = 2, \Delta x = 1/64$, $\lambda_{\max} = 1$ that's
$\Delta t \approx 10^{-3}$ — matching the values you'll see in the
examples.

## When you'd want Dormand-Prince instead

* You're running a smooth problem with strongly varying time scales,
  where most of the run can take large steps and a few regions need
  small ones. SSPRK3 with fixed $\Delta t$ pessimises for the
  smallest required step.
* You want an automatic error tolerance instead of hand-tuning
  $\Delta t$.

In both cases see [Dormand-Prince](dormand-prince.md).
