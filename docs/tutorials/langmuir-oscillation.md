# Langmuir oscillation (2D single-fluid plasma)

The Langmuir oscillation is the canonical electrostatic plasma wave —
a small density perturbation in an electron fluid against a static
ion background induces a self-restoring E-field that oscillates the
electrons at the plasma frequency. It's the simplest test that
exercises the bidirectional fluid ↔ Maxwell coupling:

* The fluid feels the Lorentz force from Maxwell's E-field.
* The fluid's current accumulates onto Maxwell's E-field RHS.

So both source terms in `SSPRK3_TwoFluid_2D` are active and you can
watch them swap energy back and forth.

## What's modelled

A periodic $[0, 1] \times [0, 1/8]$ slab. Single mobile electron
fluid with charge-to-mass ratio $\alpha = -1$ against a static positive
ion background of density 1. $\varepsilon_0 = 1$.

Initial condition: a small density perturbation
$\rho_e(x, 0) = 1 + \delta\sin(2\pi x)$ with $\delta = 0.01$. The
net charge density is $\rho_q = \alpha\rho_e + n_0 = -\delta\sin(2\pi
x)$, which by Gauss's law fixes the initial E-field:
$$
E_x(x, 0) = \frac{\delta}{k\,\varepsilon_0}\cos(2\pi x).
$$

Linearised cold-plasma theory predicts an oscillation at
$\omega_{pe} = \sqrt{\alpha^2 n_0/\varepsilon_0} = 1$, so the period
is $T = 2\pi / \omega_{pe} = 2\pi$. The example integrates for two
full periods (13 frames).

## Run it

```fish
mkdir -p /tmp/langmuir
pixi run mojo run -I src examples/langmuir_2d_vtu.mojo
```

The driver prints integrated diagnostics each frame:

```text
time         ∫ρ_e dV          ∫E_x² dV         rel Δmass
-----------  ---------------  ---------------  ------------
0.0          0.125            1.58e-7          0.0
1.05         0.125            3.97e-8          −3.6e-14
2.09         0.125            3.89e-8          −7.5e-14
3.14         0.125            1.54e-7          −1.3e-13
4.19         0.125            3.62e-8          −1.7e-13
...
```

$\int E_x^2$ oscillates between $\sim 1.6 \times 10^{-7}$ (E-field
peak — half-periods $\pi, 2\pi, 3\pi, ...$) and $\sim 4 \times 10^{-8}$
(kinetic energy peak — quarter-periods between). Mass drift stays
below $10^{-12}$ relative over the 12 500 SSPRK3 steps.

Open `/tmp/langmuir/langmuir_fluid.pvd` and
`/tmp/langmuir/langmuir_maxwell.pvd` in ParaView side-by-side. They
share the same timeline, so scrubbing one moves both.

## Walkthrough

### The closure-based plasma stepper

The `SSPRK3_TwoFluid_2D` stepper takes a single comptime
`compute_joint_rhs_fn` closure that, given the two current states
(fluid + Maxwell), writes the combined hyperbolic + source RHS for
each:

```mojo
def compute_joint_rhs(
    mut qf_in: DGState2D[P, EULER_NC],
    mut qm_in: DGState2D[P, MAXWELL_NC],
    mut qf_out: DGState2D[P, EULER_NC],
    mut qm_out: DGState2D[P, MAXWELL_NC],
) raises {imm}:
    fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
    maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
    apply_lorentz_source_2d[P](ctx, qf_in, qm_in, qf_out, ALPHA)
    apply_current_source_2d[P](ctx, qf_in, qm_out, ALPHA, EPS0)

stepper.step(ctx, q_fluid, q_maxwell, dt_use, compute_joint_rhs)
```

Inside the closure the driver picks **what** sources to apply and
**in what order**. To turn off the back-reaction (Maxwell → fluid
only), just drop the `apply_current_source_2d` call. To run with a
much weaker α, change the constant. The stepper has no opinion on
any of this.

### Why two PVD files

Maxwell and fluid live in different `DGState2D` types (different
NCs — 6 for Maxwell, 5 for the fluid). The existing VTU writers
each take one state, so the example uses two `FrameWriter`
instances:

```mojo
var fw_fluid = FrameWriter("langmuir_fluid", OUTPUT_DIR)
var fw_maxwell = FrameWriter("langmuir_maxwell", OUTPUT_DIR)
...
dump_euler_2d_frame[P](fw_fluid.next_path(t), ctx, mesh, q_fluid, GAMMA)
dump_maxwell_2d_frame[P](fw_maxwell.next_path(t), ctx, mesh, q_maxwell)
```

`FrameWriter.next_path(t)` returns the per-frame file path and
records the timestamp; `write_pvd()` at the end emits the .pvd
collection. Both writers see the same `t` sequence so the two
collections are time-synchronised.

## What you'll see

In ParaView, load `langmuir_maxwell.pvd` and colour by `E_x`. You'll
see a sinusoid that flips sign every half-period — the E-field
restoring the displaced electrons, then reversing as the electrons
overshoot.

Then load `langmuir_fluid.pvd` and colour by `rho`. You'll see the
density perturbation breathe back and forth, 90° out of phase with
the E-field (the field accelerates the electrons; the velocity
displaces them; the displacement makes the field).

The `u` field (x-velocity, recovered as $m_x/\rho$ by the VTU writer)
shows where the kinetic energy sits at each phase.

## Variations

* **Heavier ions**: this example uses a static ion background. For
  a real electron-ion plasma with finite mass ratio, set up two
  species with $\alpha_e = -1, \alpha_i = +1/m_i$ where $m_i$ is the
  ion-to-electron mass ratio. With $m_i = 1836$, $\alpha_i \approx
  5\times 10^{-4}$ and the ion contribution to $\omega_{pe}$ is
  negligible.
* **Larger amplitude**: bump $\delta$ to 0.1. The wave becomes
  weakly nonlinear; harmonics appear in ∫E_x². Beyond ~0.3 the
  fluid pressure floor activates and the wave breaks.
* **Pair plasma**: see [Roadmap](../contributing/index.md#roadmap)
  — pair plasma in 2D needs the three-state `SSPRK3_Plasma_2D`
  stepper with antisymmetric perturbations on the two species.

## Related

* [`tests/test_langmuir_2d.mojo`](../physics/plasma.md) — the
  regression-test version of this same setup.
* [`tests/test_langmuir_pair_plasma_2d.mojo`](../physics/plasma.md)
  — the two-species variant ($\omega = \sqrt{2}\omega_{pe}$).
* [`tests/test_pair_plasma_asym_scaling.mojo`](../physics/plasma.md)
  — diagnoses the O(δ²) antisymmetry residual in pair plasma.
