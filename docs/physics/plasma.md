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
from warpxm.spatial.lorentz_source_2d import apply_lorentz_source_2d
from warpxm.spatial.maxwell_current_source_2d import apply_current_source_2d

var stepper = SSPRK3_Plasma_2D[P](ctx, mesh.num_elements)

@parameter
def compute_joint_rhs(
    mut q1_in: DGState2D[P, EULER_NC],
    mut q2_in: DGState2D[P, EULER_NC],
    mut qm_in: DGState2D[P, MAXWELL_NC],
    mut q1_out: DGState2D[P, EULER_NC],
    mut q2_out: DGState2D[P, EULER_NC],
    mut qm_out: DGState2D[P, MAXWELL_NC],
) raises:
    fluid_spatial.compute_rhs(ctx, mesh, q1_in, q1_out)
    fluid_spatial.compute_rhs(ctx, mesh, q2_in, q2_out)
    maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
    apply_lorentz_source_2d[P](ctx, q1_in, qm_in, q1_out, ALPHA_1)
    apply_lorentz_source_2d[P](ctx, q2_in, qm_in, q2_out, ALPHA_2)
    apply_current_source_2d[P](ctx, q1_in, qm_out, ALPHA_1, EPS0)
    apply_current_source_2d[P](ctx, q2_in, qm_out, ALPHA_2, EPS0)

stepper.step[compute_joint_rhs](ctx, q_s1, q_s2, q_maxwell, dt)
```

The plasma stepper takes a single comptime `compute_joint_rhs_fn`
closure that, given the three current states, writes the full
hyperbolic + source RHS for each. The closure body is where the
driver picks which sources to apply, in what order, and with what
α / ε₀ values. The stepper applies SSPRK3 combine weights on each
state independently. See [Time integration](../concepts/time-integration.md)
for the rationale.

`SSPRK3_TwoFluid_{1,2,3}D` is the 2-state variant (one fluid + Maxwell)
with the same closure pattern shrunk to four buffer args.

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
| `tests/test_langmuir_1d.mojo` | 1D single-fluid Langmuir oscillation against a static positive-ion background, $\omega = \omega_{pe}$. Returns to IC after one period; mass drift ~2.7e-13. |
| `tests/test_langmuir_2d.mojo` | 2D single-fluid Langmuir oscillation, same wave $\omega = \omega_{pe}$. |
| `tests/test_langmuir_pair_plasma_1d.mojo` | 1D two-species pair plasma, $\omega = \sqrt{\alpha_{1}^{2} + \alpha_{2}^{2}}\,\omega_{pe}$. Mass drift ~1.9e-13 per species. |
| `tests/test_langmuir_pair_plasma_2d.mojo` | 2D pair plasma; validates the symmetric Lorentz / current coupling. |
| `tests/test_langmuir_pair_plasma_3d.mojo` | 3D pair plasma. |
| `tests/test_cyclotron_2d.mojo` | Single-fluid cyclotron orbit in a uniform $B_z$. Energy conserved to $\Delta E \approx 2 \times 10^{-14}$ over one period (magnetic force does no work). |
| `tests/test_cyclotron_3d.mojo` | 3D cyclotron, $\Delta E \approx 3 \times 10^{-13}$ over one period. |

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

The stepper evaluates all source terms at each RK stage rather than
operator-splitting them. The pair-plasma test's antisymmetry residual
of 1.85e-3 is **physical** (O(δ²) Lorentz coupling between
opposite-charge species with antisymmetric ρ perturbations), not
algorithmic — confirmed by the δ-scaling diagnostic in
`tests/test_pair_plasma_asym_scaling.mojo` (ratio 100.0 between
successive δ values).
