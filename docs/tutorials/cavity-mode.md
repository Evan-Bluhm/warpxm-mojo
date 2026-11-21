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
mkdir -p /tmp/maxwell_cavity
pixi run mojo run -I src examples/maxwell_cavity_2d_vtu.mojo
```

Per-frame output:

```text
frame    time        ∫E_energy dV     ∫B_energy dV     ∫total dV
-----  ----------  ----------------  ----------------  ----------------
0      0.0         0.125             0.0               0.125
1      0.118       0.0938            0.0312            0.125
2      0.236       0.0313            0.0937            0.125
3      0.354       6.5e-9            0.125             0.125
4      0.471       0.0312            0.0938            0.125
...
```

The total energy stays at 0.125 to ~10⁻⁸ across the whole period —
that's the Maxwell solver's energy conservation, not anything we add.

## Walkthrough

### PEC walls

```mojo
var mesh = BlockMesh2D[P](
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
host[b + IDX_EZ] = sin(PI * x) * sin(PI * y)
# everything else = 0
ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

var app = Maxwell(c_light=1.0, cfl_max=0.5)
var spatial = DGMaxwellSolver2D[P](app)
var stepper = SSPRK3_2D[P, MAXWELL_NC](ctx, mesh.num_elements)

def compute_rhs(
    mut q_in: DGState2D[P, MAXWELL_NC],
    mut q_out: DGState2D[P, MAXWELL_NC],
) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def no_adjust(mut q_: DGState2D[P, MAXWELL_NC]) raises {imm}:
    pass
```

No limiter is needed — Maxwell in vacuum has no shocks, so `no_adjust`
is a no-op and the stepper does pure SSPRK3.

### Diagnostics

```mojo
var e_energy = sum(½(E_x² + E_y² + E_z²))   # GLL-quadrature integrated
var b_energy = sum(½(B_x² + B_y² + B_z²))
```

These give the picture you want: in a TM mode the E and B fields
oscillate **out of phase** (E peaks at $t = 0, T/2, T$; B peaks at
$t = T/4, 3T/4$). The total $E + B$ energy stays constant.

## What you'll see in ParaView

Open `/tmp/maxwell_cavity/cavity.pvd`. Colour by:

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
