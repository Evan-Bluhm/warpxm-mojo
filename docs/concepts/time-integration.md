# Time integration

warpxm-mojo ships two explicit Runge-Kutta integrators and a family of
**source-coupled** and **adjuster-coupled** stepper variants for the
problems that need them.

## SSPRK3 — the workhorse

[Three-stage Shu-Osher SSPRK3](../numerical-methods/ssprk3.md) is the
default for almost everything. It's strong-stability-preserving (any
property the forward-Euler step preserves — positivity, TVD bounds —
SSPRK3 also preserves) and third-order accurate in time.

```mojo
from warpxm.temporal.ssprk3 import SSPRK3
var stepper = SSPRK3[P, NC](ctx, mesh.num_elements)
stepper.step(ctx, mesh, spatial, q, dt)
```

The generic `SSPRK3[P, NC]` is for 1D problems with a single state
field. 2D / 3D and physics-specific variants live alongside it in
`temporal/`:

| Stepper | Used by |
| ------- | ------- |
| `SSPRK3[P, NC]` | 1D advection, generic |
| `SSPRK3_Euler_Limited_1D/2D/3D` | Compressible Euler with shock limiter |
| `SSPRK3_Euler_Positive_1D/2D/3D` | Above + Zhang-Shu positivity preserver |
| `SSPRK3_Maxwell_2D/3D` | Maxwell, no limiter needed |
| `SSPRK3_IMHD_Limited_2D/3D` | Ideal MHD with limiter |
| `SSPRK3_IMHD_Positive_2D/3D` | MHD limiter + positivity preserver |
| `SSPRK3_IMHD_GLM_Limited_2D/3D` | MHD with Dedner GLM cleaning |
| `SSPRK3_Plasma_2D/3D` | Two-species plasma (Euler × 2 + Maxwell) |

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

DP45 currently exists for generic 1D fields and for 2D Euler. Other
physics modules use SSPRK3.

## Coupling sources and limiters

When time-integrating systems with sources (gravity, Lorentz force) or
shape-preserving constraints (slope limiter, positivity), we have a
choice about *where* in the SSPRK3 update to apply them.

### Source terms

The plasma stepper (`SSPRK3_Plasma_2D/3D`) accumulates the source into
the RHS of each RK stage:

```mojo
spatial.compute_rhs(ctx, mesh, u, rhs)        # ∇·F only
apply_lorentz_source_2d(..., u, m, rhs, ...)  # adds J×B / ρ to rhs
apply_current_source_2d(..., u, m_rhs, ...)   # adds J onto Maxwell's ∂E/∂t
```

Because each RHS evaluation is fed the current stage values, this gives
a coupled, second-order-accurate treatment of the source — see the pair-
plasma Langmuir regression test for the demonstration.

### Limiters and positivity preservers

Slope limiters and the Zhang-Shu positivity preserver are *projection*
operators on the state, not RHS modifiers. They're applied **between RK
stages**:

```mojo
# Pseudocode for one SSPRK3 step with limiter + positivity
u_save = u
spatial.compute_rhs(ctx, mesh, u_save, rhs)
stage1 = u_save + dt * rhs
limiter.apply(ctx, mesh, stage1)
positivity.apply(ctx, mesh, stage1)
spatial.compute_rhs(ctx, mesh, stage1, rhs)
stage2 = 0.75 * u_save + 0.25 * (stage1 + dt * rhs)
limiter.apply(ctx, mesh, stage2)
positivity.apply(ctx, mesh, stage2)
# ... and so on for stage 3
```

Both adjusters are **cell-mean preserving**, so mass / momentum /
energy stay conserved while ρ and P are bounded away from zero.

See [Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md)
and [Zhang-Shu positivity preserver](../numerical-methods/positivity.md)
for the algorithms.

## Choosing a stepper for your driver

A rough decision tree:

```text
Smooth problem (advection, sound wave, smooth Euler)
    → SSPRK3[P, NC]  (or DormandPrince45_1D if you want adaptive)

Compressible Euler with shocks (Sod, blast wave)
    → SSPRK3_Euler_Limited_Nd  (NX_d = 1, 2, or 3)
    → SSPRK3_Euler_Positive_Nd  if shocks are strong enough to drive ρ or P negative

Maxwell-only (cavity, plane wave)
    → SSPRK3_Maxwell_2D / 3D

Ideal MHD (Brio-Wu, Orszag-Tang, Alfvén waves)
    → SSPRK3_IMHD_Limited_2D / 3D
    → use DGImhdSolver2D[P, USE_HLLD=True] inside it for sharper shocks
    → SSPRK3_IMHD_Positive_2D / 3D for strong shocks near vacuum

MHD with divergence cleaning
    → SSPRK3_IMHD_GLM_Limited_2D / 3D

Plasma (two-species)
    → SSPRK3_Plasma_2D / 3D
```

Each of these has at least one regression test exercising it — see
[`tests/`](https://github.com/your-fork/warpxm-mojo/tree/main/tests).
