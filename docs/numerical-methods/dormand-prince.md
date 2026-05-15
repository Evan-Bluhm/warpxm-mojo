# Dormand-Prince adaptive RK45

Seven-stage embedded Runge-Kutta with a fifth-order solution and a
fourth-order error estimate. The error estimate drives an adaptive
$\Delta t$ controller.

Citation: Dormand, J. R. & Prince, P. J., "A family of embedded
Runge-Kutta formulae", *J. Comput. Appl. Math.* 6 (1980) 19–26.

## Butcher tableau

The seven stages are evaluated at fractions $c$ of the interval:

$$
c = \begin{pmatrix}0 & \tfrac{1}{5} & \tfrac{3}{10} & \tfrac{4}{5} & \tfrac{8}{9} & 1 & 1\end{pmatrix},
$$

and the fifth-order solution at $t + \Delta t$ uses weights $b$:

$$
\mathbf{u}^{n+1} = \mathbf{u}^{n} + \Delta t \!\left(
\tfrac{35}{384}k_1 + \tfrac{500}{1113}k_3 + \tfrac{125}{192}k_4 - \tfrac{2187}{6784}k_5 + \tfrac{11}{84}k_6\right).
$$

The embedded fourth-order solution uses weights $b^{\ast}$ with one
extra $k_7$ evaluation. The difference $\mathbf{y}_5 - \mathbf{y}_4 =
\Delta t (e_1 k_1 + e_3 k_3 + e_4 k_4 + e_5 k_5 + e_6 k_6 + e_7 k_7)$
with the standard coefficients

$$
e = \begin{pmatrix}\tfrac{71}{57600} & 0 & -\tfrac{71}{16695} & \tfrac{71}{1920} & -\tfrac{17253}{339200} & \tfrac{22}{525} & -\tfrac{1}{40}\end{pmatrix}.
$$

## Adaptive controller

A standard PI-free controller:

1. Compute the scaled RMS error norm

   $$
   \text{err} = \sqrt{\frac{1}{N}\sum_i \!\left(\frac{|y_{5,i} - y_{4,i}|}{\text{atol} + \text{rtol}\cdot\max(|u_n^{i}|, |y_5^{i}|)}\right)^{2}}.
   $$

2. Compute the new step size

   $$
   \Delta t_{\text{new}} = \Delta t \cdot \text{clip}(0.9\cdot \text{err}^{-1/5},\; [0.2,\, 5]).
   $$

3. If $\text{err} \leq 1$: accept the step, return $\Delta t_{\text{new}}$
   as the next guess.
4. If $\text{err} > 1$: reject, restore $\mathbf{u}^{n}$ from the
   snapshot, retry with the smaller $\Delta t_{\text{new}}$.

## API

```mojo
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
# or DormandPrince45_Euler_2D for 2D Euler

var stepper = DormandPrince45_1D[P, NC](ctx, mesh.num_elements)

# Fixed-step (same dt every call):
stepper.step(ctx, mesh, spatial, q, dt)

# Adaptive (dt floats):
var ret = stepper.step_adaptive(
    ctx, mesh, spatial, q, dt_try,
    atol=Real(1.0e-8), rtol=Real(1.0e-6),
)
var dt_used = ret[0]
var dt_next = ret[1]
```

A typical adaptive loop:

```mojo
var t: Real = 0.0
var dt = DT_INIT
while t < T_FINAL - 1e-12:
    var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
    var ret = stepper.step_adaptive(ctx, mesh, spatial, q, dt_try,
                                     atol=ATOL, rtol=RTOL)
    t += ret[0]
    dt = ret[1]
```

## What you get

Compared to fixed-Δt SSPRK3 at the same final error level:

| Test | Steps (SSPRK3, dt=5e-4) | Steps (DP45 adaptive) |
| ---- | ----------------------- | --------------------- |
| 1D advection smooth, P=3, T=1 | ~2000 | **187** |
| 2D Euler entropy wave smooth, T=1 | ~2000 | **199** |

A ~10× saving on smooth problems. On problems with sharp transitions
the controller pulls back to a smaller $\Delta t$ near gradients and
relaxes elsewhere — savings are smaller but you also don't have to
hand-tune $\Delta t$.

## Coverage

| Module | Status |
| ------ | ------ |
| `dormand_prince_1d.mojo` (generic 1D `[P, NC]`) | ✅ fixed + adaptive |
| `dormand_prince_euler_2d.mojo` | ✅ fixed + adaptive |
| 2D / 3D MHD, Maxwell, plasma | Pending |

The DP45 architecture is dimension-agnostic — the combine kernels in
`dormand_prince_1d.mojo` operate on flat buffers and are reused
unchanged by the 2D Euler variant. Porting to other 2D / 3D physics is
roughly a 100-line copy of the existing 2D Euler driver.

## Implementation notes

* Mojo's aliasing checker forbids passing the same buffer to multiple
  `MutAnyOrigin` arguments. The combine step that would naively take
  "up to 7 contribution buffers" is split into six specialised
  kernels with 1, 2, 3, 4, and 5 (twice — used at two different
  stages) contribution buffers.
* The host-side error reduction (RMS across all GPU buffer elements)
  is done by mapping the err / u / u_save buffers to host and looping.
  For very large meshes this becomes a bottleneck — a future
  optimisation is a GPU-side reduction kernel.
