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
pixi run example euler_sedov_blast_2d_vtu
```

Per-frame output (values rounded):

```text
frame  time  ∫ρ dV  ∫E dV    max|u|  min ρ   min P_gas  rel Δmass  rel ΔE   CFL dt
0      0.0   1.0    0.07318  0       1.0     1.0e-3     0          0        2.09e-4
1      0.01  1.0    0.07318  1.173   0.554   1.0e-3     4.1e-14    1.4e-14  2.26e-4
2      0.02  1.0    0.07318  1.141   0.329   6.8e-4     5.0e-14    1.1e-14  2.64e-4
5      0.05  1.0    0.07318  0.860   0.135   1.0e-10    5.1e-14    2.4e-14  3.80e-4
10     0.1   1.0    0.07318  0.633   0.084   1.0e-10    7.9e-14    3.0e-14  4.31e-4
```

What to notice:

* `min P_gas` reaches the floor `p_eps = 1e-10` from $t = 0.05$ on. That
  is the preserver clamping; without it the run fails with $P < 0$.
* `min ρ` falls to 0.084 in the rarefied core, well above its floor.
* `rel Δmass` and `rel ΔE` stay below $10^{-13}$: the limiter and the
  preserver both conserve cell means.
* `CFL dt` doubles over the run, from 2.1e-4 to 4.3e-4, as the blast
  expands and the peak sound speed drops.

Frames and `sedov.pvd` are written to `output/sedov/`.

## Walkthrough

### Limiter, positivity preserver and closures

```mojo
var app = Euler(
    gas_gamma=GAMMA, min_density=Real(1e-12),
    min_pressure=Real(1e-12), cfl_max=Real(0.5),
)
var spatial = DGEulerSolver[2, P](app)
var limiter = MoeRossmanith[2, P, EULER_NC](
    ctx, mesh.num_elements, alpha_coeff=Real(100.0),
)
var positivity = PositivityEuler[2, P](
    num_elements=mesh.num_elements, gamma=GAMMA,
    rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
)
var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)
var fw = FrameWriter("sedov")

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)

def suggest_dt(mut q_: State) raises {imm} -> Real:
    return spatial.suggest_dt(ctx, mesh, q_)
```

The `adjust` closure runs `limiter.apply(...)` then
`positivity.apply(...)` after each RK stage. The order matters:

* The **limiter** bounds each element's variation by its neighbours'
  means. It conserves cell means but does not guarantee positivity.
* The **positivity preserver** acts only where the limiter's output
  still violates $\rho > 0$ or $P_{\text{gas}} > 0$ at a node.

See [Zhang-Shu positivity preserver](../numerical-methods/positivity.md)
for why this conserves.

### Why `alpha_coeff = 100`

The limiter ignores deviations smaller than
$\alpha = \texttt{alpha\_coeff} \cdot \Delta x^{1.5}$. With the default
coefficient of 1000, $\alpha \approx 2$ on this mesh, larger than
anything in the problem, so the limiter would do nothing and the
rarefied core would be held up by the positivity floor alone. Combined
with a CFL-driven step that produces a collapse: an evacuating node
raises the sound speed, which shrinks the step, which lets the node
evacuate further (the step fell from 2e-5 to 3e-8 within 40 steps, at
every `cfl_max` tried). `alpha_coeff = 100` gives $\alpha \approx 0.2$,
which keeps the limiter active on the blast without flattening it. See
[Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md).

### Frame loop and time step

```mojo
var t: Real = 0.0
for frame in range(NUM_FRAMES):
    dump_euler_frame(fw.next_path(t), mesh, q, GAMMA)

    var max_u: Real = 0.0
    var min_rho: Real = 1.0e30
    var min_p: Real = 1.0e30
    def extrema(
        x: Real, y: Real, w: Real, v: Array[Real, EULER_NC],
    ) {mut max_u, mut min_rho, mut min_p}:
        var rho = v[IDX_RHO]
        var ux = v[IDX_MX] / rho
        var uy = v[IDX_MY] / rho
        var ke = Real(0.5) * rho * (ux * ux + uy * uy)
        max_u = fmax(max_u, sqrt(ux * ux + uy * uy))
        min_rho = fmin(min_rho, rho)
        min_p = fmin(min_p, (GAMMA - Real(1.0)) * (v[IDX_E] - ke))
    mesh.visit_nodes(q, extrema)
    var mass = mesh.integrate(q, IDX_RHO)
    var e_int = mesh.integrate(q, IDX_E)
    # ... print ...

    if frame < NUM_FRAMES - 1:
        t = stepper.run_until_cfl(
            ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
            compute_rhs, adjust, suggest_dt,
        )
```

`mesh.visit_nodes` calls the closure with the coordinates, quadrature
weight and state of every node; `mut` captures accumulate the extrema.
`run_until_cfl` re-evaluates the step from $\max(|u| + c)$ every step
(`dt_interval` defaults to 1), which suits a problem whose peak speed
changes this fast. Pass `dt_min` to make it raise if the suggested step
falls to or below a threshold, or is NaN, rather than grinding on.

### Floors

```mojo
PositivityEuler[2, P](
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

Open `output/sedov/sedov.pvd`. Colour by `rho`:

* $t = 0$: tight Gaussian pressure spike in the centre, uniform $\rho$.
* $t = 0.02$: spherical shock front at $r \approx 0.1$.
* $t = 0.1$: shock at $r \approx 0.4$, with a low-density rarefaction
  cavity at the centre.

The expanding shock and the evacuated core are both resolved; the pulse
is a finite-width Gaussian, so this is Sedov-like rather than the exact
point-source similarity solution.

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
