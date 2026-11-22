# Rusanov / Lax-Friedrichs flux

The default numerical flux used at every element interface in
warpxm-mojo. Robust, dissipative, and well-suited to a wide range of
shocks.

## Formula

Given left and right states $\mathbf{q}_L, \mathbf{q}_R$ at an
interface with unit normal $\hat{n}$, the Rusanov flux is

$$
\mathbf{F}^{\ast}(\mathbf{q}_L, \mathbf{q}_R; \hat{n}) =
\tfrac{1}{2}\!\left(\mathbf{F}(\mathbf{q}_L)\cdot\hat{n} + \mathbf{F}(\mathbf{q}_R)\cdot\hat{n}\right)
- \tfrac{1}{2}\,\lambda_{\max}\,(\mathbf{q}_R - \mathbf{q}_L),
$$

where $\lambda_{\max}$ is an upper bound on the maximum signal speed at
the interface. The choice of $\lambda_{\max}$ depends on the physics:

| Physics | $\lambda_{\max}$ |
| ------- | ---------------- |
| Advection | $\max(|v_n|_L, |v_n|_R)$ (or just upwind directly) |
| Euler | $\max\!\left(|u_n|_L + c_{s,L},\, |u_n|_R + c_{s,R}\right)$, $c_s = \sqrt{\gamma P / \rho}$ |
| Maxwell | $c$ (speed of light) |
| MHD | $\max\!\left(|u_n|_L + c_{f,L},\, |u_n|_R + c_{f,R}\right)$, $c_f$ the fast magnetosonic speed |

Fast magnetosonic for MHD:

$$
c_f^{2} = \tfrac{1}{2}\!\left(c_s^{2} + c_A^{2}\right) +
          \tfrac{1}{2}\sqrt{(c_s^{2} + c_A^{2})^{2} - 4\,c_s^{2}\,(c_A^{n})^{2}},
$$

with $c_A^{2} = |\mathbf{B}|^{2} / \rho$, $c_A^{n} = B_n/\sqrt{\rho}$.

## Pros and cons

**Why it's the default**

* Works for every physics module without special-casing.
* Always stable for $\lambda_{\max}\Delta t / \Delta x < 1$.
* Simple to derive (just an entropy fix on an averaged centered flux).

**When you want something better**

* For strong MHD shocks, Rusanov smears the contact and Alfvén
  discontinuities badly. Switch to [HLLD](hlld.md) instead — same
  signature, much sharper.
* For very strong Euler shocks (P-ratios in the thousands) the
  dissipation can drive intermediate states unphysical. Stack the
  [Zhang-Shu positivity preserver](positivity.md) on top.

## In the code

Every physics module exposes `numerical_flux_static(...)` as the
Rusanov implementation. Example from `apps/imhd.mojo`:

```mojo
@staticmethod
def numerical_flux_static(
    q_l: Array[Real, MHD_NC],
    q_r: Array[Real, MHD_NC],
    nx: Real, ny: Real, nz: Real,
    gamma: Real, rho_min: Real, press_min: Real,
) -> Array[Real, MHD_NC]:
    var fs_l = _normal_flux_and_speed(q_l, nx, ny, nz, gamma, rho_min, press_min)
    var fs_r = _normal_flux_and_speed(q_r, nx, ny, nz, gamma, rho_min, press_min)
    var c_max = fmax(fs_l[MHD_NC], fs_r[MHD_NC])

    var f: Array[Real, MHD_NC] = Array[Real, MHD_NC](fill=Real(0.0))
    comptime for c in range(MHD_NC):
        f[c] = Real(0.5) * (fs_l[c] + fs_r[c]) - Real(0.5) * c_max * (q_r[c] - q_l[c])
    return f^
```

`_normal_flux_and_speed` returns $\mathbf{F}\cdot\hat{n}$ in slots
`0..NC-1` and the signal speed $|u_n| + c_f$ in slot `NC`. It forms the
normal flux directly, one evaluation per side, instead of building the
three axis fluxes and dotting them with the normal.

1D callers pass `(±1, 0, 0)` for the normal; 2D callers pass `(nx, ny, 0)`.
The unused lanes multiply by zero and the compiler folds them away.

DG kernels call this once per face per RK stage.
