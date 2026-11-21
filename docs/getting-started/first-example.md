# Running your first example

We'll run `examples/advection_1d.mojo` — the smallest end-to-end driver
in the repo — and walk through what it does. By the end you'll know how
the four core pieces (**mesh**, **state**, **spatial solver**,
**stepper**) fit together. Every more advanced example uses the same
pattern.

## Run it

```fish
pixi run mojo run -I src examples/advection_1d.mojo
```

Expected output:

```text
Initial q (first nodes):
  q[ 0 ] = -0.5
  q[ 1 ] = -0.20710678118654752
  ...

Integrated to t = 1.0 in 1000 steps
Final q (first nodes):
  q[ 0 ] = -0.5000000000000001
  ...
L2 error vs IC after one period: 2.36e-07
```

The L2 error of ~10⁻⁷ confirms the DG solver returned the sine wave to
its initial state after one full period, as expected for `v = 1` on a
unit-length periodic domain.

## What the file says

Every example is a single Mojo program that imports the solver modules
and calls into them — there is no separate config file. The
`advection_1d.mojo` driver has four sections, mirrored in essentially
every example.

### 1. Compile-time configuration

```mojo
comptime P: Int = 2                  # polynomial order (3 GLL nodes / elem)
comptime NUM_ELEMENTS: Int = 64
comptime X_MIN: Real = -0.5
comptime X_MAX: Real = 0.5
comptime VELOCITY: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime DT: Real = 1.0e-3
```

These are **compile-time constants** (Mojo's `comptime`). Changing them
re-compiles the driver. There is no runtime input file — to vary a
parameter, edit and re-run. This trade-off buys aggressive
specialisation: the polynomial order `P` is known to the compiler, so
DG kernels unroll their per-element loops with no runtime branches.

### 2. Build the mesh + state

```mojo
var ctx = DeviceContext()
var mesh = BlockMesh1D[P](ctx, X_MIN, X_MAX, NUM_ELEMENTS, periodic=True)
var q = DGState1D[P, 1](ctx, mesh.num_elements)
init_sine(ctx, q, mesh)
```

* `DeviceContext` is Mojo's GPU handle. One per process.
* `BlockMesh1D[P]` is a uniform 1D mesh with periodic or per-side
  boundary conditions. The `P` parameter (polynomial order) is baked in
  so the mesh knows the node count per element (`P + 1`).
* `DGState1D[P, NC]` is the conserved-variable storage. The second
  parameter `NC` is the number of components — `1` for scalar
  advection, `5` for 5-moment Euler, `8` for ideal MHD, etc.

`init_sine` fills `q` with sin(2π x) on the mesh's GLL nodes. The
write goes through a host buffer and an `enqueue_copy` to the device.

### 3. Set up physics + spatial solver + stepper

```mojo
var app = Advection[1](velocity_x=VELOCITY, cfl_max=Real(1.0))
var spatial = DGSpatialSolver1D[P, 1](app)
var stepper = SSPRK3_1D[P, 1](ctx, mesh.num_elements)
```

* **App** describes the physics — for advection that's just the wave
  speed. Other apps (`Euler`, `Maxwell`, `IMHD`, …) carry their own
  parameters like γ-law indices or speed of light. Each physics has a
  single canonical struct shared across all dimensions; the per-dim DG
  kernel supplies the right normal triples.
* **Spatial solver** wraps the app and knows how to compute the DG
  spatial RHS (volume + face fluxes). Each (physics, dimension)
  combination has its own spatial solver type.
* **Stepper** is the time integrator. `SSPRK3_1D[P, NC]` is the generic
  three-stage Shu-Osher SSPRK3 scheme suitable for any single-state
  problem in 1D (`SSPRK3_2D[P, NC]` and `SSPRK3_3D[P, NC]` for higher
  dimensions). It takes two comptime closures — `compute_rhs_fn` and
  `adjust_fn` — at the call site so the driver composes physics RHS,
  source terms, limiters, and positivity preservers freely. See
  [Time integration](../concepts/time-integration.md).

### 4. Time-step loop

```mojo
def compute_rhs(
    mut q_in: DGState1D[P, 1],
    mut q_out: DGState1D[P, 1],
) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

def no_adjust(mut q_: DGState1D[P, 1]) raises {imm}:
    pass

var t: Real = 0.0
while t < T_FINAL - Real(0.5) * DT:
    var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
    stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
    t += dt_use
```

Boilerplate — closures capture `spatial`/`mesh`/`ctx`; the time loop
calls `step(ctx, q, dt, compute_rhs, no_adjust)`. For adaptive RK45
swap in `DormandPrince45_1D[P, NC]` and call
`stepper.step_adaptive(..., compute_rhs)` (see
[Dormand-Prince](../numerical-methods/dormand-prince.md)).

## Try changing something

A few cheap experiments to get a feel for the system:

* Set `P = 1` (slope-only DG, two GLL nodes per element). The L2 error
  jumps by ~10× — that's the 2nd-order vs 3rd-order convergence rate.
* Set `NUM_ELEMENTS = 32` — the L2 error grows by 2³ = 8 (3rd-order
  spatial convergence).
* Set `VELOCITY = 0.7`. After `T_FINAL = 1`, the wave has only travelled
  70 % of a period; the final state is `sin(2π(x - 0.7))`, not the IC.

## Next

* See [Sod tube tutorial](../tutorials/sod-tube.md) for the same
  recipe applied to compressible Euler with shocks.
* Or jump to [Concepts → Design departures](../concepts/design-departures.md)
  for the why behind the compile-time config + GPU-only model.
