# Sod tube (Euler)

The Sod shock tube is the canonical 1D compressible-flow benchmark, with
a closed-form Riemann solution. This tutorial runs
`examples/euler_sod_2d_vtu.mojo` — the 2D-extruded form with VTU output
so you can watch the rarefaction, contact, and shock evolve in
ParaView.

## What's modelled

Initial condition:

| Region | $\rho$ | $u$ | $P$ |
| ------ | ------ | --- | --- |
| Left  ($x < 0.5$) | 1.0   | 0 | 1.0 |
| Right ($x \geq 0.5$) | 0.125 | 0 | 0.1 |

$\gamma = 7/5$, copyout boundary in $x$, free-slip walls in $y$.
$t_{\text{final}} = 0.2$. The solution at $t > 0$ contains, from left
to right: a left-going rarefaction fan, a contact discontinuity, and a
right-going shock.

## Run it

```fish
pixi run example euler_sod_2d_vtu
```

The driver writes one snapshot of the final state at $t = 0.2$ to
`output/sod_2d.vtu` and prints the time reached, the last CFL step and
the total mass. Open the file in ParaView (see
[Viewing output in ParaView](../getting-started/paraview.md)). For a
multi-frame run with `FrameWriter` and a `.pvd` collection, see the
[Orszag-Tang tutorial](orszag-tang.md) or the
[Langmuir tutorial](langmuir-oscillation.md).

## Walkthrough

### Setup

```mojo
from warpxm.prelude import *
from warpxm.physics.euler import *

comptime State = DGState[2, P, EULER_NC]

var mesh = BlockMesh[2, P](
    ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
    nx=NX, ny=NY, periodic_x=False, periodic_y=False,
    bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
)
var q = State(ctx, mesh.num_elements)
```

The mesh is **anisotropic**: `NX = 200` cells in $x$ where the physics
lives, `NY = 8` cells in $y$ only so the VTU writer has something to
draw. The $y$ boundaries are free-slip walls, so the solution stays
1D-extruded.

### Initial condition

One closure returning the conserved state at a point; `mesh.fill`
evaluates it at every GLL node:

```mojo
def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
    var rho: Real = 1.0 if x < Real(0.5) else 0.125
    var pp: Real = 1.0 if x < Real(0.5) else 0.1
    return [rho, Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
mesh.fill(q, initial)
```

### Solver, limiter and closures

```mojo
var app = Euler(
    gas_gamma=GAMMA, min_density=Real(1e-12),
    min_pressure=Real(1e-12), cfl_max=Real(0.5),
)
var spatial = DGEulerSolver[2, P](app)
var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements)

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def adjust(mut q_: State) raises {mut limiter, imm}:
    limiter.apply(ctx, mesh, q_)

def suggest_dt(mut q_: State) raises {imm} -> Real:
    return spatial.suggest_dt(ctx, mesh, q_)
```

The `adjust` closure runs after each RK stage; see
[Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md). The
Sod problem is not strong enough to need positivity preservation. For
pressure ratios above about 100, add a `PositivityEuler[2, P]` and call
`positivity.apply(ctx, mesh, q_)` inside `adjust`.

### Time stepping

```mojo
var t = stepper.run_until_cfl(
    ctx, q, Real(0.0), T_FINAL, compute_rhs, adjust, suggest_dt,
    dt_interval=5,
)

makedirs(OUTPUT_DIR, exist_ok=True)
var path = OUTPUT_DIR + "/sod_2d.vtu"
dump_euler_frame(path, mesh, q, GAMMA)
```

There is no hard-coded time step. `spatial.suggest_dt` reduces
$\max(|u| + c)$ over all nodes on the GPU and returns
`cfl_max / ((2P + 1) · λ · (1/Δx + 1/Δy))`; with `cfl_max = 0.5` that is
about `1.6e-4` here. `run_until_cfl` asks for a new step every
`dt_interval` steps (each request costs one reduction and a host
read-back) and shortens the last step to land on `T_FINAL`. It also
takes `dt_min`: if the suggested step is at or below it, or is NaN, the
call raises instead of stalling.

`dump_euler_frame` writes $
ho$, $u$, $v$, $w$, $P$. For intermediate
frames use a `FrameWriter` and call `fw.next_path(t)` at each frame
time; see [`io`](../api/io.md).

## Expected result

After $t = 0.2$:

| $x$ | $\rho$ (analytic) |
| --- | ----------------- |
| 0.1 | 1.0 |
| 0.6 | 0.426 |
| 0.75 | 0.266 |
| 0.95 | 0.125 |

The numerical solution at the same grid points matches to about 3 × 10⁻³,
limited by the DG-with-limiter dissipation. Compare with
`tests/test_euler_sod_2d.mojo` for the exact tolerances.

## In ParaView

1. Open `output/sod_2d.vtu`.
2. Apply.
3. Color by `rho`.
4. **Filters → Data Analysis → Plot Over Line** with endpoints at
   $(0, 0.05, 0)$ and $(1, 0.05, 0)$ gives the classic 1D Sod plot.

By $t = 0.2$ the shock has propagated to $x \approx 0.86$.

## Variations to try

* Increase pressure ratio to 100:1 — at some point you'll start seeing
  negative pressure failures. Add `PositivityEuler[2, P]` to your `adjust`
  closure (see the [Sedov blast tutorial](sedov-blast.md)).
* Change $\gamma$ to 2.0 — analogous to the Brio-Wu MHD setup, useful if
  you want to compare hydrodynamic vs magnetised shock structure.
* Refine the mesh to $\text{NX} = 400$ — the shock front becomes
  noticeably sharper, especially with HLLD on the MHD side.
