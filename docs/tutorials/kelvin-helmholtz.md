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
pixi run example euler_kelvin_helmholtz_2d_vtu
# → 11 frames + kh.pvd in output/kh/
```

The driver runs on a $128^{2}$ mesh at $P = 2$ to $T = 2$. The CFL step
is about `3e-4` (roughly 6700 SSPRK3 steps); the run takes about 20
seconds on a desktop GPU and prints integrated mass and energy at each
of the 11 frames.

Open `output/kh/kh.pvd` in ParaView and colour by `rho`. By frames 4 to
6 ($t \approx 1.0$) the two shear layers visibly roll into a row of
vortices; by frame 10 they have started to merge.

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

ρ and $u_x$ are both interpolated through this shape, inside the
closure handed to `mesh.fill`:

```mojo
def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
    var S = _shape(y)
    var rho = RHO_OUTER + (RHO_INNER - RHO_OUTER) * S
    var ux = U_OUTER + (U_INNER - U_OUTER) * S
    # v-perturbation localised near the two interfaces.
    var bump1 = exp(-(y - Real(0.25)) * (y - Real(0.25)) / (SIGMA * SIGMA))
    var bump2 = exp(-(y - Real(0.75)) * (y - Real(0.75)) / (SIGMA * SIGMA))
    var uy = DELTA_V * sin(Real(2.0) * TWO_PI * x) * (bump1 + bump2)
    var ke = Real(0.5) * rho * (ux * ux + uy * uy)
    return [
        rho, rho * ux, rho * uy, Real(0.0),
        PRESSURE / (GAMMA - Real(1.0)) + ke,
    ]
mesh.fill(q, initial)
```

The Gaussian-localised perturbation is restricted to the interface
regions, so the $u_y$ kick is not spent on parts of the flow that do not
shear.

### Stepper and time step

```mojo
var app = Euler(
    gas_gamma=GAMMA, min_density=Real(1e-12),
    min_pressure=Real(1e-12), cfl_max=Real(1.0),
)
var spatial = DGEulerSolver[2, P](app)
var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements)
var fw = FrameWriter("kh")

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def adjust(mut q_: State) raises {mut limiter, imm}:
    limiter.apply(ctx, mesh, q_)

def suggest_dt(mut q_: State) raises {imm} -> Real:
    return spatial.suggest_dt(ctx, mesh, q_)
```

and, per frame interval,

```mojo
t = stepper.run_until_cfl(
    ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
    compute_rhs, adjust, suggest_dt, dt_interval=10,
)
```

The standard Euler driver pattern: no sources, the limiter as `adjust`,
and a CFL-driven step. The flow is smooth, so the example sets
`cfl_max = 1.0` rather than the 0.5 used for shock problems; the
suggested step stays between 2.8e-4 and 3.1e-4. The sound speed barely
changes, so the step is refreshed only every 10 steps
(`dt_interval=10`). `run_until_cfl` also takes `dt_min` and raises if
the suggested step falls to or below it, or is NaN.

KH at this density ratio does not need positivity preservation. If you
raise the density ratio, or run until the vortex sheets thin enough to
undershoot, add a `PositivityEuler[2, P]` to the `adjust` closure.

## What you'll see

After each frame the driver prints integrated mass and energy and the
current CFL step:

```text
time   ∫ρ dV   ∫E dV   rel Δmass   rel ΔE   CFL dt
0.0  1.49999999995  3.91875234998   0.0       0.0       3.07e-4
0.2  1.49999999995  3.91875234998  -3.4e-13  -2.9e-13   3.07e-4
...
2.0  1.49999999995  3.91875234998  -6.3e-13  -5.3e-13   2.80e-4
```

Mass and energy drift are about $6 \times 10^{-13}$ relative at
$t = 2$, after roughly 6700 steps of fully nonlinear evolution: round-off
in the conservative update, nothing more.

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
