# Kelvin-Helmholtz instability (Euler 2D)

The Kelvin-Helmholtz instability is the canonical "two parallel streams
shearing past each other" hydrodynamic instability. A small
perpendicular perturbation grows exponentially and rolls the shear
layer into vortices. It's a classic test that:

* Has no shock — so it cleanly stresses the DG accuracy on a
  smooth-but-stiff problem.
* Runs for long time (~2 dynamical periods) — exercises mass / energy
  conservation under sustained nonlinear evolution.
* Demands a **smooth** initial condition. A jump discontinuity at the
  shear interface would dominate the early-time evolution with Gibbs
  ringing; we use `tanh` smoothing to make the IC $C^\infty$.

## What's modelled

A periodic $[0, 1]^2$ domain split into three horizontal bands:

| Region | $\rho$ | $u_x$ |
| ------ | ------ | ----- |
| Outer ($y < 0.25$ and $y > 0.75$) | 1 | $-0.5$ |
| Inner ($0.25 < y < 0.75$) | 2 | $+0.5$ |

Uniform pressure $P_0 = 2.5$, $\gamma = 5/3$. The transitions at
$y = 0.25$ and $y = 0.75$ are smoothed with a `tanh` shape of width
$\sigma = 0.025$, so the IC is everywhere differentiable. The
$y$-velocity perturbation is a single-mode Gaussian-localised sine:

$$
u_y(x, y) = \delta_v \cdot \sin(4\pi x) \cdot
            \left[\exp\!\left(-\frac{(y - 0.25)^{2}}{\sigma^{2}}\right) +
                  \exp\!\left(-\frac{(y - 0.75)^{2}}{\sigma^{2}}\right)\right]
$$

with $\delta_v = 0.01$. This seeds the two shear layers in phase.

## Run it

```fish
mkdir -p /tmp/kh
pixi run mojo run -I src examples/euler_kelvin_helmholtz_2d_vtu.mojo
# → 11 frames + .pvd at /tmp/kh
```

The driver runs on a $128^{2}$ mesh at $P = 2$ for $T = 2$. At
$\Delta t = 10^{-3}$ that's 2000 SSPRK3 steps; it takes a couple of
minutes on a desktop GPU and prints integrated mass and energy at each
of the 11 frames.

Open `/tmp/kh/kh.pvd` in ParaView and colour by `rho`. By frame 4-6
($t \approx 1.0$) the two shear layers visibly roll into a row of
vortices; by frame 10 they've started to merge.

## Walkthrough

### The smooth IC

The interesting math is in the smoothing trick:

```mojo
@always_inline
def _shape(y: Real) -> Real:
    """Smooth indicator for the inner layer: ~1 on (0.25, 0.75), ~0 outside."""
    return Real(0.5) * (tanh((y - Real(0.25)) / SIGMA)
                        - tanh((y - Real(0.75)) / SIGMA))
```

$S(y)$ is essentially a $C^\infty$ "rectangular pulse" that's ≈ 1 in
$(0.25, 0.75)$ and ≈ 0 outside. With $\sigma = 0.025$ on a $128^2$
mesh ($\Delta x \approx 0.008$) the transition is resolved by ~3
elements — small enough to look like a discontinuity at the eye but
smooth enough that the DG polynomial basis can represent it without
Gibbs ringing.

ρ and $u_x$ are both interpolated through this shape:

```mojo
var S = _shape(y)
var rho = RHO_OUTER + (RHO_INNER - RHO_OUTER) * S
var ux = U_OUTER + (U_INNER - U_OUTER) * S
```

The Gaussian-localised perturbation is restricted to the interface
regions so the $u_y$ kick doesn't waste momentum on the parts of the
flow that don't actually shear.

### Stepper

```mojo
var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)

def compute_rhs(mut q_in, mut q_out) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def adjust(mut q_: DGState2D[P, EULER_NC]) raises {mut limiter, imm}:
    limiter.apply(ctx, mesh, q_)

stepper.step(ctx, q, dt_use, compute_rhs, adjust)
```

Standard Euler driver pattern — no sources, just the limiter as
`adjust`. KH is smooth enough that you don't need positivity
preservation; if you crank up the density ratio or run longer until
vortex sheets get thin enough to underflow, add a
`PositivityEuler2D` to the `adjust` closure.

## What you'll see

After each frame the driver prints integrated mass and energy:

```text
time         ∫ρ dV            ∫E dV            rel Δmass     rel ΔE
-----------  ---------------  ---------------  ------------  ------------
0.0   1.5  3.918...   0.0   0.0
0.2   1.5  3.918...   −3.6e−13   −2.8e−13
...
2.0   1.5  3.918...   −5.0e−13   −4.0e−13
```

Mass and energy drift sit at $\sim 5 \times 10^{-13}$ relative after
2000 SSPRK3 steps — essentially machine epsilon. The flow is fully
non-linear by then, but conservation holds exactly (modulo
floating-point round-off in the GLL quadratures).

## Variations

* **Coarser** runs (`NX = NY = 64`): twice as fast, vortex rollup
  still visible but with less fine structure.
* **Larger Δρ**: bump `RHO_INNER` to 5 or 10. The instability grows
  faster but underresolved cells may underflow — switch on the
  positivity preserver.
* **Stronger shear**: bump `|U_OUTER - U_INNER|`. Watch the
  pressure-jump shock turn this from KH into a Rayleigh-Taylor-like
  mix.
* **Different perturbation modes**: change `sin(4π x)` to
  `sin(2π x) + sin(8π x)` for nested vortices that compete and merge.
