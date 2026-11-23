# Langmuir oscillation (2D single-fluid plasma)

The Langmuir oscillation is the canonical electrostatic plasma wave —
a small density perturbation in an electron fluid against a static
ion background induces a self-restoring E-field that oscillates the
electrons at the plasma frequency. It's the simplest test that
exercises the bidirectional fluid ↔ Maxwell coupling:

* The fluid feels the Lorentz force from Maxwell's E-field.
* The fluid's current accumulates onto Maxwell's E-field RHS.

So both source terms in `SSPRK3_TwoFluid` are active and you can
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
pixi run example langmuir_2d_vtu
```

The driver prints integrated diagnostics each frame (values rounded):

```text
2D Langmuir oscillation, ω_pe = 1.0 , T = 6.283185307179586 , dt = 0.00078125
time   ∫ρ_e dV   ∫E_x² dV   rel Δmass
0.0     0.125  1.583e-7   0.0
1.047   0.125  3.963e-8  -4.0e-14
2.094   0.125  3.896e-8  -1.0e-13
3.142   0.125  1.542e-7  -1.7e-13
...
12.566  0.125  1.580e-7  -7.1e-13
```

$\int E_x^2$ oscillates between about $1.6 \times 10^{-7}$ (field
energy peak, every half period) and about $4 \times 10^{-8}$ in between.
After two periods it is back to 1.580e-7 from 1.583e-7. Mass drift stays
below $10^{-12}$ relative over the roughly 16 000 SSPRK3 steps.

Open `output/langmuir/langmuir_fluid.pvd` and
`output/langmuir/langmuir_maxwell.pvd` in ParaView side by side. They
share the same timeline, so scrubbing one moves both.

## Walkthrough

### Initial condition

Two states on the same mesh, each filled from its own closure:

```mojo
comptime Fluid = DGState[2, P, EULER_NC]
comptime Field = DGState[2, P, MAXWELL_NC]

def fluid_initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
    return [
        Real(1.0) + DELTA * sin(TWO_PI * x), Real(0.0), Real(0.0), Real(0.0),
        P0 / (GAMMA - Real(1.0)),
    ]

def field_initial(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EX] = (DELTA / (TWO_PI * EPS0)) * cos(TWO_PI * x)
    return v^
mesh.fill(q_fluid, fluid_initial)
mesh.fill(q_maxwell, field_initial)
```

`warpxm.physics.plasma` exports both apps, their solvers, the two
coupling sources and the joint steppers.

### The closure-based plasma stepper

`SSPRK3_TwoFluid[NDIM, P]` takes one `compute_joint_rhs` closure that,
given the two current states (fluid and Maxwell), writes the combined
hyperbolic + source right-hand side for each:

```mojo
var stepper = SSPRK3_TwoFluid[2, P](ctx, mesh.num_elements)

def compute_joint_rhs(
    mut qf_in: Fluid, mut qm_in: Field, mut qf_out: Fluid, mut qm_out: Field,
) raises {imm}:
    fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
    maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
    apply_lorentz_source(ctx, qf_in, qm_in, qf_out, ALPHA)
    apply_current_source(ctx, qf_in, qm_out, ALPHA, EPS0)
```

Inside the closure the driver picks **which** sources to apply and
**in what order**. To turn off the back-reaction (Maxwell → fluid
only), drop the `apply_current_source` call. The stepper has no opinion
on any of this.

### Time step

```mojo
var dt = fmin(
    maxwell_spatial.suggest_dt(mesh),
    fluid_spatial.suggest_dt(ctx, mesh, q_fluid),
)
...
t = stepper.run_until(
    ctx, q_fluid, q_maxwell,
    t, t_final * Real(frame + 1) / Real(NUM_FRAMES - 1), dt,
    compute_joint_rhs,
)
```

The step is fixed for the run, taken as the smaller of the two CFL
steps at $t = 0$. The coupled stepper has `run_until` but no CFL-driven
variant, and nothing here changes the limit: light ($c = 1$) is far
faster than the cold electron fluid, so the Maxwell step
(`7.8125e-4` with `cfl_max = 0.5`) bounds the hyperbolic part, and it
also resolves the plasma frequency that the source terms introduce
($\omega_{pe}\,\Delta t \approx 10^{-3}$). For a denser plasma, check
that $\omega_{pe}\,\Delta t$ stays well below 1: the source terms are
explicit and the CFL estimate does not see them.

### Why two PVD files

Maxwell and fluid live in different `DGState` types (different
NCs: 6 for Maxwell, 5 for the fluid). A VTU file holds one state, so
the example uses two `FrameWriter` instances pointed at the same
directory:

```mojo
comptime OUTPUT_DIR: String = "output/langmuir"

var fw_fluid = FrameWriter("langmuir_fluid", OUTPUT_DIR)
var fw_maxwell = FrameWriter("langmuir_maxwell", OUTPUT_DIR)
...
dump_euler_frame(fw_fluid.next_path(t), mesh, q_fluid, GAMMA)
dump_maxwell_frame(fw_maxwell.next_path(t), mesh, q_maxwell)
```

`FrameWriter.next_path(t)` returns the per-frame file path and
records the timestamp; `write_pvd()` at the end emits the .pvd
collection. The directory is created by the first writer. Both writers see the same `t` sequence so the two
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
* **Pair plasma**: two mobile species need the three-state
  `SSPRK3_Plasma[NDIM, P]` stepper with antisymmetric perturbations on
  the two species; `tests/test_langmuir_pair_plasma_2d.mojo` is a
  working setup.

## Related

* [`tests/test_langmuir_2d.mojo`](../physics/plasma.md) — the
  regression-test version of this same setup.
* [`tests/test_langmuir_pair_plasma_2d.mojo`](../physics/plasma.md)
  — the two-species variant ($\omega = \sqrt{2}\omega_{pe}$).
* [`tests/test_pair_plasma_asym_scaling_2d.mojo`](../physics/plasma.md)
  — diagnoses the O(δ²) antisymmetry residual in pair plasma.
