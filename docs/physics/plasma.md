# Two-fluid plasma

Two species of charged fluid coupled to Maxwell's equations through the
Lorentz force and the displacement current. Each species is its own
5-moment Euler fluid; the shared Maxwell field carries the
electromagnetic forces and receives the species' currents.

## Equations

For each species $s$ with charge-to-mass ratio $\alpha_s$:

$$
\partial_t
\begin{pmatrix}\rho_s \\ \rho_s\mathbf{u}_s \\ E_s\end{pmatrix}
+
\nabla\cdot
\begin{pmatrix}\rho_s\mathbf{u}_s \\ \rho_s\mathbf{u}_s\mathbf{u}_s + P_s\,I \\ (E_s + P_s)\mathbf{u}_s\end{pmatrix}
=
\begin{pmatrix}0 \\ \alpha_s\,\rho_s(\mathbf{E} + \mathbf{u}_s\times\mathbf{B}) \\ \alpha_s\,\rho_s\mathbf{u}_s\cdot\mathbf{E}\end{pmatrix},
$$

and Maxwell receives the current from each species:

$$
\partial_t \mathbf{E} = c^{2}\nabla\times\mathbf{B} - \frac{1}{\varepsilon_{0}}\sum_{s} \alpha_s\rho_s\mathbf{u}_s,
\qquad
\partial_t \mathbf{B} = -\nabla\times\mathbf{E}.
$$

A "single-fluid plasma" is just the two-fluid setup with one species.

## Stepper

```mojo
from warpxm.temporal.ssprk3_plasma_2d import SSPRK3_Plasma_2D
from warpxm.temporal.ssprk3_plasma_3d import SSPRK3_Plasma_3D

var stepper = SSPRK3_Plasma_2D[P](ctx, mesh.num_elements)
stepper.step(
    ctx, mesh, fluid_spatial, maxwell_spatial,
    q_s1, q_s2, q_maxwell,
    alpha_s1, alpha_s2, eps0, dt,
)
```

The plasma stepper bundles three states (two fluids + Maxwell). Inside
each RK stage:

1. Compute the spatial RHS of each species and of Maxwell separately.
2. Apply the **Lorentz source** to each species' momentum / energy RHS
   using the current Maxwell state.
3. Accumulate each species' **current** onto Maxwell's E-RHS.
4. Combine into the next stage with the standard SSPRK3 weights.

This is a fully coupled (not split) integration — see
[Time integration](../concepts/time-integration.md) for the rationale.

## Source kernels

* `apply_lorentz_source_2d/_3d` — $\alpha\rho(\mathbf{E} + \mathbf{u}\times\mathbf{B})$
  on the species momentum, $\alpha\rho\mathbf{u}\cdot\mathbf{E}$ on the
  species energy.
* `apply_current_source_2d/_3d` — $-\alpha\rho\mathbf{u}/\varepsilon_0$
  on Maxwell's E-field RHS.

Both are NC-parameterised so they reuse the same kernel across plasma
species.

## Examples and tests

| Test | Setup |
| ---- | ----- |
| `tests/test_langmuir_2d.mojo` | Single-fluid Langmuir oscillation. $\omega = \omega_{pe}$ — the wave returns to its IC after one period. |
| `tests/test_langmuir_pair_plasma_2d.mojo` | Two species with opposite charge, equal mass and density. $\omega = \sqrt{\alpha_{1}^{2} + \alpha_{2}^{2}}\,\omega_{pe}$. Validates the symmetric Lorentz / current coupling. |
| `tests/test_langmuir_pair_plasma_3d.mojo` | 3D analog of the above. |
| `tests/test_cyclotron_2d.mojo` | Single-fluid cyclotron orbit in a uniform $B_z$. Energy conserved to 4 × 10⁻¹⁵ (magnetic force does no work). |
| `tests/test_cyclotron_3d.mojo` | 3D cyclotron. |

There is no 1D plasma — the lowest-rank Langmuir oscillation needs a
real Maxwell field which is at least 2D.

## Choosing α

* **Pair plasma** (electron-positron): $\alpha_1 = -1, \alpha_2 = +1$,
  equal mass.
* **Electron-ion plasma**: $\alpha_e \approx -1$, $\alpha_i \approx
  +1/m_{i/e}$ — for protons that's roughly $+1/1836$. The ion
  contribution to the plasma frequency is then negligible and the
  system reduces to a single-species Langmuir oscillation around an
  effectively static ion background.

Use whatever charge / mass / density normalisations are convenient;
the equations of motion are linear in $\alpha$.

## Time-integration accuracy

The current source-coupled stepper is a fully-coupled RK3 evaluation
of all source terms at each stage. There's a pending task to add
symmetric Strang splitting for cases where stiff sources dominate;
see [Roadmap](../contributing/index.md#roadmap).
