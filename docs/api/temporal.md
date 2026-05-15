# `warpxm.temporal`

Explicit time integrators. Two families: SSPRK3 (three-stage Shu-Osher
strong-stability-preserving) and Dormand-Prince 5(4).

## SSPRK3 variants

| File | Stepper struct | Use |
| ---- | -------------- | --- |
| `ssprk3.mojo` | `SSPRK3[P, NC]` | Generic 1D, no adjuster |
| `ssprk3_euler_limited_1d.mojo` | `SSPRK3_Euler_Limited_1D[P]` | 1D Euler + limiter |
| `ssprk3_euler_limited_2d.mojo` | `SSPRK3_Euler_Limited_2D[P]` | 2D Euler + limiter |
| `ssprk3_euler_limited_3d.mojo` | `SSPRK3_Euler_Limited_3D[P]` | 3D Euler + limiter |
| `ssprk3_euler_positive_1d.mojo` | `SSPRK3_Euler_Positive_1D[P]` | Above + positivity preserver |
| `ssprk3_euler_positive_2d.mojo` | `SSPRK3_Euler_Positive_2D[P]` | … |
| `ssprk3_euler_positive_3d.mojo` | `SSPRK3_Euler_Positive_3D[P]` | … |
| `ssprk3_maxwell_2d.mojo` | `SSPRK3_Maxwell_2D[P]` | 2D Maxwell |
| `ssprk3_maxwell_3d.mojo` | `SSPRK3_Maxwell_3D[P]` | 3D Maxwell |
| `ssprk3_imhd_limited_2d.mojo` | `SSPRK3_IMHD_Limited_2D[P]` | 2D MHD + limiter |
| `ssprk3_imhd_limited_3d.mojo` | `SSPRK3_IMHD_Limited_3D[P]` | 3D MHD + limiter |
| `ssprk3_imhd_positive_2d.mojo` | `SSPRK3_IMHD_Positive_2D[P]` | Above + MHD positivity preserver |
| `ssprk3_imhd_positive_3d.mojo` | `SSPRK3_IMHD_Positive_3D[P]` | … |
| `ssprk3_imhd_glm_limited_2d.mojo` | `SSPRK3_IMHD_GLM_Limited_2D[P]` | 2D GLM-MHD + limiter |
| `ssprk3_imhd_glm_limited_3d.mojo` | `SSPRK3_IMHD_GLM_Limited_3D[P]` | 3D GLM-MHD + limiter |
| `ssprk3_plasma_2d.mojo` | `SSPRK3_Plasma_2D[P]` | Two-species plasma (Euler × 2 + Maxwell) |
| `ssprk3_plasma_3d.mojo` | `SSPRK3_Plasma_3D[P]` | … |

### Generic API

```mojo
from warpxm.temporal.ssprk3 import SSPRK3

var stepper = SSPRK3[P, NC](ctx, mesh.num_elements)
stepper.step(ctx, mesh, spatial, q, dt)
```

### Limited / positive Euler 2D

```mojo
var spatial = DGEulerSolver2D[P](app)
var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)
var stepper = SSPRK3_Euler_Limited_2D[P](ctx, mesh.num_elements)
stepper.step(ctx, mesh, spatial, limiter, q, dt)

# Positivity version: same plus a PositivityEuler2D.
var positivity = PositivityEuler2D[P](num_elements=..., gamma=...)
var stepper = SSPRK3_Euler_Positive_2D[P](ctx, mesh.num_elements)
stepper.step(ctx, mesh, spatial, limiter, positivity, q, dt)
```

### MHD with HLLD

```mojo
var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
var stepper = SSPRK3_IMHD_Limited_2D[P](ctx, mesh.num_elements)
stepper.step[USE_HLLD=True](ctx, mesh, spatial, limiter, q, dt)
```

The `[USE_HLLD=True]` comptime parameter is forwarded so the stepper
internally calls the HLLD flux at every face evaluation.

### Plasma 2D

```mojo
var stepper = SSPRK3_Plasma_2D[P](ctx, mesh.num_elements)
stepper.step(
    ctx, mesh, fluid_spatial, maxwell_spatial,
    q_s1, q_s2, q_maxwell,
    alpha_s1, alpha_s2, eps0, dt,
)
```

## Dormand-Prince 5(4)

```mojo
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
from warpxm.temporal.dormand_prince_euler_2d import DormandPrince45_Euler_2D

var stepper = DormandPrince45_1D[P, NC](ctx, mesh.num_elements)
# or DormandPrince45_Euler_2D[P]

# Fixed-step:
stepper.step(ctx, mesh, spatial, q, dt)

# Adaptive:
var ret = stepper.step_adaptive(ctx, mesh, spatial, q, dt_try,
                                atol=Real(1.0e-8), rtol=Real(1.0e-6))
var dt_used = ret[0]
var dt_next = ret[1]
```

Currently shipped for 1D generic and 2D Euler. See
[Dormand-Prince](../numerical-methods/dormand-prince.md) for the
algorithm; porting to other 2D / 3D physics is ~100 lines per module.

## Internal helpers

The two basic combination kernels live in `ssprk3.mojo`:

```mojo
def _euler_step_kernel(...)  # out[i] = x[i] + dt * rhs[i]
def _ssprk3_combine_kernel(...)  # out[i] = a*u0[i] + b*(u_stage[i] + dt*rhs[i])
```

Both are dimension-agnostic (flat 1D buffers) and reused by every
SSPRK3 variant. Six specialised combine kernels for DP45 (1-, 2-, 3-,
4-, 5-input arities) live in `dormand_prince_1d.mojo`.
