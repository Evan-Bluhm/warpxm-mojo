# Running your first example

We'll run `examples/advection_1d.mojo` — the smallest end-to-end driver
in the repo — and walk through what it does. By the end you'll know how
the four core pieces (**mesh**, **state**, **spatial solver**,
**stepper**) fit together. Every more advanced example uses the same
pattern.

## Run it

```fish
pixi run example advection_1d
```

Expected output:

```text
Initial q (first nodes):
  q[ 0 ] = -1.2246467991473532e-16
  q[ 1 ] = -0.049067674327417966
  ...

Integrated to t = 1.0 with dt = 0.0009375
Final q (first nodes):
  q[ 0 ] = 3.811733169446261e-05
  ...
L2 error vs IC after one period: 1.2747864163420074e-05
```

The L2 error of about 10⁻⁵ confirms the DG solver returned the sine wave
to its initial state after one full period, as expected for `v = 1` on a
unit-length periodic domain.

## What the file says

Every example is a single Mojo program that imports the solver modules
and calls into them — there is no separate config file. The
`advection_1d.mojo` driver has five parts, which recur in every example.

### 1. Imports and compile-time configuration

```mojo
from warpxm.prelude import *
from warpxm.physics.advection import *

comptime P: Int = 2                  # polynomial order (3 GLL nodes / elem)
comptime NUM_ELEMENTS: Int = 64
comptime X_MIN: Real = -0.5
comptime X_MAX: Real = 0.5
comptime VELOCITY: Real = 1.0
comptime CFL: Real = 0.3
comptime T_FINAL: Real = 1.0         # one full period
comptime State = DGState[1, P, 1]
```

`warpxm.prelude` brings in everything that is not tied to one physics
(`Real`, the meshes, `DGState`, the steppers, output helpers);
`warpxm.physics.advection` adds the app and its solvers. See
[prelude and physics](../api/prelude.md).

The constants are **compile-time** (Mojo's `comptime`). Changing them
re-compiles the driver. There is no runtime input file: to vary a
parameter, edit and re-run. In exchange the polynomial order `P` is
known to the compiler, so DG kernels unroll their per-element loops with
no runtime branches. There is no time step in the list: it comes from
the CFL number (step 4).

### 2. Build the mesh and state, set the initial condition

```mojo
var ctx = DeviceContext()
var mesh = BlockMesh[1, P](ctx, X_MIN, X_MAX, NUM_ELEMENTS, periodic=True)
var q = State(ctx, mesh.num_elements)

def initial(x: Real) -> Array[Real, 1]:
    return [sin(TWO_PI * x)]
mesh.fill(q, initial)
```

* `DeviceContext` is Mojo's GPU handle. One per process.
* `BlockMesh[1, P]` is a uniform 1D mesh with periodic or per-side
  boundary conditions. The polynomial order `P` is a type parameter, so
  the mesh knows the node count per element (`P + 1`).
* `DGState[NDIM, P, NC]` is the conserved-variable storage: `NDIM`
  spatial dimensions, `NC` components per node (`1` for scalar
  advection, `5` for 5-moment Euler, `8` for ideal MHD).
* `mesh.fill(q, initial)` evaluates the closure at every GLL node and
  writes the result into `q`. The closure takes the node coordinates
  (`x` in 1D, `x, y` in 2D, `x, y, z` in 3D) and returns all `NC`
  components. You never index the state buffer yourself.

### 3. Set up physics, spatial solver and stepper

```mojo
var app = Advection[1](velocity_x=VELOCITY, cfl_max=CFL)
var spatial = DGAdvectionSolver[1, P, 1](app)
var stepper = SSPRK3[1, P, 1](ctx, mesh.num_elements)
```

* **App** describes the physics. For advection that is the velocity and
  the CFL number. Other apps (`Euler`, `Maxwell`, `IMHD`, …) carry their
  own parameters, such as the adiabatic index or the speed of light.
  Each physics has one struct shared across all dimensions.
* **Spatial solver** wraps the app and computes the DG right-hand side
  (volume and face fluxes). Each (physics, dimension) pair has its own
  solver type.
* **Stepper** is the time integrator. `SSPRK3[NDIM, P, NC]` is the
  three-stage Shu-Osher scheme for any single-state problem. It takes
  two closures, `compute_rhs` and `adjust`, so the driver composes the
  physics right-hand side, source terms, limiters and positivity
  preservers freely. See
  [Time integration](../concepts/time-integration.md).

### 4. Advance in time

```mojo
def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

# The velocity is uniform, so the CFL step is fixed for the whole run.
var dt = spatial.suggest_dt(mesh)
var t = stepper.run_until(
    ctx, q, Real(0.0), T_FINAL, dt, compute_rhs, no_op_adjust[1, P, 1],
)
```

The closure captures `spatial`, `mesh` and `ctx` read-only (`{imm}`).
`suggest_dt` returns the largest stable step for the app's `cfl_max`:
`cfl_max / ((2P + 1) |v| / Δx)`, here `0.3 / (5 · 64) = 9.375e-4`.
`run_until` takes steps of `dt` from `t = 0` to `T_FINAL`, shortening the
last one to land on the target exactly, and returns the time reached.
`no_op_adjust` is the "no limiter" adjuster.

For problems whose wave speed depends on the solution (Euler, MHD), use
`suggest_dt(ctx, mesh, q)` and `stepper.run_until_cfl(...)`, which
re-evaluates the step as the run proceeds; the
[Sod tube tutorial](../tutorials/sod-tube.md) shows it. For adaptive
RK45 use `DormandPrince45[1, P, NC]` and `step_adaptive` (see
[Dormand-Prince](../numerical-methods/dormand-prince.md)).

### 5. Diagnostics

```mojo
def exact(x: Real) -> Real:
    return sin(TWO_PI * x)
print("L2 error vs IC after one period:", mesh.l2_error(q, 0, exact))
```

`mesh.l2_error(q, comp, exact)` is `sqrt(∫ (q[comp] − exact)² dV)` by GLL
quadrature. `mesh.integrate`, `mesh.integrate_fn` and `mesh.visit_nodes`
cover the other common diagnostics; see [mesh](../api/mesh.md).

## Try changing something

A few cheap experiments to get a feel for the system:

* Set `P = 1` (two GLL nodes per element). The L2 error rises by two
  orders of magnitude: the scheme drops from third to second order.
* Set `NUM_ELEMENTS = 32`. The L2 error grows by about 2³ = 8
  (third-order convergence). `examples/convergence_advection_1d.mojo`
  measures the rate for P = 1, 2, 3.
* Set `VELOCITY = 0.7`. After `T_FINAL = 1`, the wave has only travelled
  70 % of a period; the final state is `sin(2π(x - 0.7))`, not the IC.

## Next

* See [Sod tube tutorial](../tutorials/sod-tube.md) for the same
  recipe applied to compressible Euler with shocks.
* Or jump to [Concepts → Design departures](../concepts/design-departures.md)
  for the why behind the compile-time config + GPU-only model.
