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
from warpxm.temporal.dormand_prince_2d import DormandPrince45_2D
# or DormandPrince45_1D / DormandPrince45_3D for other dimensions

var stepper = DormandPrince45_2D[P, NC](ctx, mesh.num_elements)

@parameter
def compute_rhs(
    mut q_in: DGState2D[P, NC],
    mut q_out: DGState2D[P, NC],
) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    # add source kernels here if needed

# Fixed-step (same dt every call):
stepper.step[compute_rhs](ctx, q, dt)

# Adaptive (dt floats):
var ret = stepper.step_adaptive[compute_rhs](
    ctx, q, dt_try,
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
    var ret = stepper.step_adaptive[compute_rhs](ctx, q, dt_try,
                                                  atol=ATOL, rtol=RTOL)
    t += ret[0]
    dt = ret[1]
```

## What you get

Compared to fixed-Δt SSPRK3 at the same final error level:

| Test | Steps (SSPRK3) | Steps (DP45 adaptive) |
| ---- | -------------- | --------------------- |
| 1D advection smooth, P=3, T=1 (dt=5e-4) | ~2000 | **187** |
| 2D Euler entropy wave smooth, T=1 (dt=1e-3) | ~2000 | **200** |
| 2D MHD Alfvén wave, T=1 (dt=1e-3) | 1000 | **169** |
| 2D Maxwell plane wave, T=1/√2 (dt=5e-4) | ~1414 | **76** |
| 3D advection smooth, T=1 (dt=1e-3) | ~1000 | **64** |
| 3D MHD Alfvén wave, T=1 (dt=1e-3) | 1000 | **131** |
| 3D Maxwell plane wave, T=1 (dt=5e-4) | ~2000 | **50** |

Savings of 6–40× on smooth problems. On problems with sharp transitions
the controller pulls back near gradients and relaxes elsewhere — savings
are smaller but you don't have to hand-tune $\Delta t$.

## Coverage

| Module | Status |
| ------ | ------ |
| `dormand_prince_1d.mojo` — `DormandPrince45_1D[P, NC]` | ✅ fixed + adaptive |
| `dormand_prince_2d.mojo` — `DormandPrince45_2D[P, NC]` | ✅ fixed + adaptive |
| `dormand_prince_3d.mojo` — `DormandPrince45_3D[P, NC]` | ✅ fixed + adaptive |
| Plasma (3 coupled states) | Not in scope — plasma steppers use SSPRK3 only |

All three steppers use the same two-closure pattern
(`step[compute_rhs_fn]`, `step_adaptive[compute_rhs_fn]`). The combine
kernels `_combine1`–`_combine5`, the per-cell error kernel
`_dp45_error_kernel`, and the single-block error-norm reduction
`_dp45_sum_sq_kernel` all live in `dormand_prince_1d.mojo` and are
imported unchanged by the 2D and 3D variants — they operate on flat
buffers and don't care about dimension.

## Implementation notes

* Mojo's aliasing checker forbids passing the same buffer to multiple
  `MutAnyOrigin` arguments. The combine step that would naively take
  "up to 7 contribution buffers" is split into five specialised
  kernels `_combine1`–`_combine5` with 1, 2, 3, 4, and 5 contribution
  buffers respectively. `_combine5` is dispatched twice per step
  (once for k6, once for the final u^{n+1}); the other four are
  called exactly once.
* The RMS scaled error norm is computed fully on the GPU.
  `_dp45_sum_sq_kernel` (in `dormand_prince_1d.mojo`, shared by the
  2D and 3D wrappers) launches one block of 256 threads that
  stride-loops over the flat buffer, tree-reduces `(err/sc)²` in
  shared memory, and writes the total to a one-element device
  buffer. The host then maps one scalar — no `map_to_host` of the
  full state, err, or u_save buffers. On large meshes this is
  several × faster than the original three-buffer host-side loop;
  on small meshes (~10⁴ DOFs) the controller is fast enough either
  way and the win is mainly in cleaner architecture.
