# TM cavity mode (Maxwell)

A 2D rectangular cavity with perfectly-conducting walls supports a
standing-wave TM₁₁ mode:

$$
E_z(x, y, t) = \sin(\pi x)\sin(\pi y)\cos(\omega t),
$$

with $\omega = c\pi\sqrt{2}$ and period $T = \sqrt{2}/c$. After one
period the fields return to the IC. This tutorial runs
`examples/maxwell_cavity_2d_vtu.mojo` and explains how PEC walls drive
the standing-wave structure.

## What's modelled

* Domain: $[0, 1]^2$, $\text{NX} = \text{NY} = 32$, $P = 2$.
* IC: $E_z = \sin(\pi x)\sin(\pi y)$ at $t = 0$; all other components 0.
* BCs: PEC walls on all four sides.
* Runs one full period in 13 output frames.

## Run it

```fish
pixi run example maxwell_cavity_2d_vtu
```

Per-frame output (values rounded):

```text
CFL dt = 0.0015625
frame   time   ∫E_energy dV   ∫B_energy dV   ∫total dV
0   0.0     0.12500000  0.0         0.12500000
1   0.1179  0.09375001  0.03124999  0.12500000
2   0.2357  0.03125002  0.09374997  0.12499999
3   0.3536  2.4e-11     0.12499999  0.12499999
...
12  1.4142  0.12499997  4.9e-11     0.12499997
Wrote 13 frames and output/maxwell_cavity/cavity.pvd
```

The total energy drops from 0.125 by 3.5e-8 over the period. That is
the dissipation of the upwind flux at this resolution; nothing is added
to conserve it.

## Walkthrough

### PEC walls

```mojo
var mesh = BlockMesh[2, P](
    ctx, 0.0, 1.0, 0.0, 1.0, nx=NX, ny=NY,
    periodic_x=False, periodic_y=False,
    bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
    bc_bottom=BC_PEC_WALL, bc_top=BC_PEC_WALL,
)
```

In the BC adjuster the ghost-state at a PEC wall has tangential E
flipped to its negative — so the average across the boundary is zero,
which is what the conducting-wall boundary condition demands.

### IC and stepper

```mojo
def initial(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EZ] = sin(PI * x) * sin(PI * y)
    return v^
mesh.fill(q, initial)

var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
var spatial = DGMaxwellSolver[2, P](app)
var stepper = SSPRK3[2, P, MAXWELL_NC](ctx, mesh.num_elements)
var fw = FrameWriter("cavity", "output/maxwell_cavity")

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
```

No limiter is needed: Maxwell in vacuum has no shocks, so the stepper is
given `no_op_adjust[2, P, MAXWELL_NC]`.

### Time step and frame loop

```mojo
var t_final = sqrt(Real(2.0)) / C_LIGHT
var dt = spatial.suggest_dt(mesh)

var t: Real = 0.0
for frame in range(NUM_FRAMES):
    dump_maxwell_frame(fw.next_path(t), mesh, q)
    var ee = mesh.integrate_fn(q, e_energy)
    var be = mesh.integrate_fn(q, b_energy)
    print(frame, t, ee, be, ee + be)

    if frame < NUM_FRAMES - 1:
        t = stepper.run_until(
            ctx, q, t, t_final * Real(frame + 1) / Real(NUM_FRAMES - 1), dt,
            compute_rhs, no_op_adjust[2, P, MAXWELL_NC],
        )
```

The light speed is constant, so the CFL step is too:
`suggest_dt(mesh)` needs no state and returns
`cfl_max / ((2P + 1) · c · (1/Δx + 1/Δy))`, which is `1.5625e-3` here.
It is computed once and `run_until` advances to each frame time with
that fixed step, shortening the last step of each interval.

### Diagnostics

```mojo
def e_energy(v: Array[Real, MAXWELL_NC]) -> Real:
    return Real(0.5) * (
        v[IDX_EX] * v[IDX_EX] + v[IDX_EY] * v[IDX_EY] + v[IDX_EZ] * v[IDX_EZ]
    )
```

and likewise `b_energy`; `mesh.integrate_fn(q, f)` integrates a
pointwise function of the node state by GLL quadrature. In a TM mode
the E and B fields oscillate **out of phase** (E peaks at
$t = 0, T/2, T$; B peaks at $t = T/4, 3T/4$) and the total stays
constant.

## What you'll see in ParaView

Open `output/maxwell_cavity/cavity.pvd`. Colour by:

* `E_z` — single peak in the centre at $t = 0$ that flips sign at
  $t = T/2$ and returns by $t = T$.
* `B_x` and `B_y` — peak at $t = T/4$ in opposing quadrants.
* `E_energy`, `B_energy` — total energy distributions.

The total energy in either field oscillates as $\cos^{2}(\omega t)$ or
$\sin^{2}(\omega t)$ respectively, with their sum constant.

## Variations

* **3D cavity**: `examples/maxwell_cavity_3d_vtu.mojo` runs the same
  setup in 3D with uniform-along-z initial condition. The TM₁₁ mode
  becomes a TM₁₁₀ mode.
* **Higher modes**: change the IC to $\sin(2\pi x)\sin(\pi y)$ and you
  get TM₂₁ with frequency $\omega' = c\pi\sqrt{5}$.
* **PMC walls**: swap `BC_PEC_WALL` → `BC_PMC_WALL`. Tangential B
  vanishes instead of tangential E — the dual TE mode emerges.
