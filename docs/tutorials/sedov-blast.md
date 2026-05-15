# Sedov blast with positivity preservation

A strong spherical pressure pulse expanding into a low-pressure
background — a hard test where the conserved-variable limiter alone
cannot keep the gas pressure positive. This tutorial runs
`examples/euler_sedov_blast_2d_vtu.mojo`, which uses the
[Zhang-Shu positivity preserver](../numerical-methods/positivity.md)
in addition to the slope limiter.

## What's modelled

* Domain: $[0, 1]^2$, periodic BCs, $\text{NX} = \text{NY} = 64$, $P = 2$.
* Ambient: $\rho = 1$, $P_{\text{amb}} = 10^{-3}$.
* Pulse: centred Gaussian with $P_{\text{inner}} = 10$, $\sigma = 0.03$.
  Pressure ratio $\boldsymbol{10^{4}}$.
* $\gamma = 7/5$, $t_{\text{final}} = 0.1$, 11 output frames.

The pressure ratio is high enough that without positivity preservation
the solver NaNs almost immediately as the shock front pulls $P_{\text{gas}}$
below zero in the rarefaction wake.

## Run it

```fish
mkdir -p /tmp/sedov_2d
pixi run mojo run -I src examples/euler_sedov_blast_2d_vtu.mojo
```

Per-frame output:

```text
frame   time      ∫ρ dV       ∫E dV       max|u|        min ρ          min P_gas      rel Δmass    rel ΔE
-----  -------  -----------  -----------  -----------  -------------  -------------  -----------  -----------
0      0.0      1.0          0.073        0            1.0            0.001          0            0
1      0.01     1.0          0.073        1.515        0.410          9.86e-11       4.3e-14      1.2e-14
2      0.02     1.0          0.073        1.428        0.183          7.45e-11       4.6e-14      9.1e-15
...
```

What to notice:

* `min P_gas` is consistently at `~10⁻¹⁰` — exactly the floor
  `p_eps`. That's the preserver clamping. Without it, the simulation
  would crash with $P < 0$.
* `min ρ` drops to ~0.01 in the rarefaction core, also above floor.
* `rel Δmass`, `rel ΔE` stay at $\sim 10^{-14}$ throughout — mass and
  total energy conserved to machine precision despite the active
  preservation.

## Walkthrough

### The positive stepper

```mojo
var spatial = DGEulerSolver2D[P](app)
var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)
var positivity = PositivityEuler2D[P](
    num_elements=mesh.num_elements, gamma=GAMMA,
    rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
)
var stepper = SSPRK3_Euler_Positive_2D[P](ctx, mesh.num_elements)

stepper.step(ctx, mesh, spatial, limiter, positivity, q, dt_use)
```

Inside `SSPRK3_Euler_Positive_2D.step`, every RK stage is followed by
`limiter.apply(...)` *then* `positivity.apply(...)`. The order matters:

* The **limiter** does shape-preservation (keep variation bounded by
  neighbour means). It guarantees conservation but not positivity.
* The **positivity preserver** kicks in only when the limiter's output
  still violates ρ > 0 or $P_{\text{gas}} > 0$ at a node.

This stacking is the standard recipe — see
[Zhang-Shu positivity preserver](../numerical-methods/positivity.md)
for the conservation proof.

### Floors

```mojo
PositivityEuler2D[P](
    num_elements=mesh.num_elements, gamma=GAMMA,
    rho_eps=Real(1.0e-10),
    p_eps=Real(1.0e-10),
)
```

The `_eps` floors aren't physical — they're "how close to zero is
acceptable before we projection-shrink toward the cell mean". Lower
values let more variation through (sharper shocks); higher values are
more dissipative. $10^{-10}$ is a good default; we've used it on every
positivity test.

## What you'll see in ParaView

Open `/tmp/sedov_2d/sedov.pvd`. Colour by `rho`:

* $t = 0$: tight Gaussian pressure spike in the centre, uniform $\rho$.
* $t = 0.02$: spherical shock front at $r \approx 0.1$.
* $t = 0.1$: shock at $r \approx 0.4$, with a low-density rarefaction
  cavity at the centre.

The full Sedov self-similar evolution is visible; the simulation
matches the analytic radial profile within the DG resolution.

## Variations

* **Push harder**: increase $P_{\text{inner}}$ to $10^{3}$ (ratio
  $10^{6}$). The preserver still keeps the run alive; mass and energy
  conservation hold.
* **Real Sedov**: replace the Gaussian with a point-energy source
  (single-cell impulse). The DG framework smooths this out into a
  Gaussian-like distribution anyway because the basis is polynomial.
* **3D**: the analogous problem in 3D is in
  `examples/euler_blast_3d_vtu.mojo` — same idea, hex sub-cells in
  the VTU output.
