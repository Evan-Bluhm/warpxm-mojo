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
pixi run mojo run -I src examples/euler_sod_2d_vtu.mojo
```

The driver writes a single snapshot of the final state at $t = 0.2$
to `/tmp/sod_2d.vtu`. Open it in ParaView (see
[Viewing output in ParaView](../getting-started/paraview.md)). For a
multi-frame run with `FrameWriter` + a `.pvd` collection, see the
[Orszag-Tang tutorial](orszag-tang.md) or the
[Langmuir tutorial](langmuir-oscillation.md).

## Walkthrough

### Setup

```mojo
var mesh = BlockMesh2D[P](
    ctx, 0.0, 1.0, 0.0, 0.1, nx=200, ny=8,
    periodic_x=False, periodic_y=False,
    bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
)
var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
```

Notice the **anisotropic mesh**: 200 cells in $x$ where the physics
lives, 8 cells in $y$ just to make the example 2D so the VTU writer
has something to draw. The y-direction has free-slip walls so the
solution stays 1D-extruded — the y-drift in $\rho$ stays at 10⁻¹³.

### IC

A standard left/right state assignment at each GLL node:

```mojo
for j in range(NY):
    for i in range(NX):
        var xl = mesh.x_min + Real(i) * mesh.dx
        for jn in range(P1):
            for in_ in range(P1):
                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                if x < 0.5:
                    rho = 1.0; pp = 1.0
                else:
                    rho = 0.125; pp = 0.1
                ...
```

### Stepper

```mojo
var app = Euler(gas_gamma=Real(1.4), ...)
var spatial = DGEulerSolver2D[P](app)
var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)

def compute_rhs(
    mut q_in: DGState2D[P, EULER_NC],
    mut q_out: DGState2D[P, EULER_NC],
) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def adjust(mut q_: DGState2D[P, EULER_NC]) raises {mut limiter, imm}:
    limiter.apply(ctx, mesh, q_)
```

The `adjust` closure runs between each RK stage — see
[Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md). The
Sod problem isn't strong enough to need positivity preservation; for
shock ratios > ~100 you'd add `positivity.apply(ctx, mesh, q_)` inside
`adjust`.

### Loop

```mojo
var t: Real = 0.0
while t < T_FINAL - 0.5 * DT:
    var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
    stepper.step(ctx, q, dt_use, compute_rhs, adjust)
    t += dt_use
ctx.synchronize()

dump_euler_2d_frame[P]("/tmp/sod_2d.vtu", ctx, mesh, q, GAMMA)
```

One straight time loop up to $t = 0.2$, then a single VTU dump. If
you want intermediate frames, swap the dump call for a `FrameWriter`
and call `fw.next_path(t)` at frame boundaries — see
[`driver.frame_writer`](../api/driver.md#framewriter).

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

1. Open `/tmp/sod_2d.vtu`.
2. Apply.
3. Color by `rho`.
4. **Filters → Data Analysis → Plot Over Line** with endpoints at
   $(0, 0.05, 0)$ and $(1, 0.05, 0)$ gives the classic 1D Sod plot.

By $t = 0.2$ the shock has propagated to $x \approx 0.86$.

## Variations to try

* Increase pressure ratio to 100:1 — at some point you'll start seeing
  negative pressure failures. Add `PositivityEuler2D` to your `adjust`
  closure (see the [Sedov blast tutorial](sedov-blast.md)).
* Change $\gamma$ to 2.0 — analogous to the Brio-Wu MHD setup, useful if
  you want to compare hydrodynamic vs magnetised shock structure.
* Refine the mesh to $\text{NX} = 400$ — the shock front becomes
  noticeably sharper, especially with HLLD on the MHD side.
