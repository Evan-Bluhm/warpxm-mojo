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
mkdir -p /tmp/sod_2d
pixi run mojo run -I src examples/euler_sod_2d_vtu.mojo
```

The driver dumps frames into `/tmp/sod_2d/sod_2d.NNNN.vtu` plus a
`sod_2d.pvd` collection. After the run you can open the PVD in
ParaView (see [Viewing output in ParaView](../getting-started/paraview.md)).

## Walkthrough

### Setup

```mojo
var mesh = BlockMesh2D[P](
    ctx, 0.0, 1.0, 0.0, 0.1, nx=200, ny=4,
    periodic_x=False, periodic_y=False,
    bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
)
var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
```

Notice the **anisotropic mesh**: 200 cells in $x$ where the physics
lives, 4 cells in $y$ just to make the example 2D so the VTU writer
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

@parameter
def compute_rhs(
    mut q_in: DGState2D[P, EULER_NC],
    mut q_out: DGState2D[P, EULER_NC],
) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

@parameter
def adjust(mut q_: DGState2D[P, EULER_NC]) raises:
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
for frame in range(NUM_FRAMES):
    dump_euler_2d_frame[P](fw.next_path(t), ctx, mesh, q, GAMMA)
    if frame < NUM_FRAMES - 1:
        var t_target = Real(frame + 1) * frame_dt
        while t < t_target - 0.5 * DT:
            var dt_use = DT if (t + DT) < t_target else (t_target - t)
            stepper.step[compute_rhs, adjust](ctx, q, dt_use)
            t += dt_use
fw.write_pvd()
```

Outer loop is over frames; inner loop is the time stepping between
frames. `FrameWriter` (`driver/frame_writer.mojo`) handles the per-
frame filenames and writes the `.pvd` collection at the end.

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

1. Open `/tmp/sod_2d/sod_2d.pvd`.
2. Apply.
3. Color by `rho`.
4. **Filters → Data Analysis → Plot Over Line** with endpoints at
   $(0, 0.05, 0)$ and $(1, 0.05, 0)$ gives the classic 1D Sod plot.

The 11 frames step through $t = 0$ to $t = 0.2$ in equal slices. By
$t \approx 0.07$ the wave structure has fully developed; by $t = 0.2$
the shock is at $x \approx 0.86$.

## Variations to try

* Increase pressure ratio to 100:1 — at some point you'll start seeing
  negative pressure failures. Add `PositivityEuler2D` to your `adjust`
  closure (see the [Sedov blast tutorial](sedov-blast.md)).
* Change $\gamma$ to 2.0 — analogous to the Brio-Wu MHD setup, useful if
  you want to compare hydrodynamic vs magnetised shock structure.
* Refine the mesh to $\text{NX} = 400$ — the shock front becomes
  noticeably sharper, especially with HLLD on the MHD side.
