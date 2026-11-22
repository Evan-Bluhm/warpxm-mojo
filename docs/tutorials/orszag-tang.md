# Orszag-Tang vortex (MHD)

The Orszag-Tang vortex is *the* canonical 2D MHD benchmark — a smooth
incompressible-looking IC that develops complex shock-interaction
structure over time. This tutorial runs the full
`examples/imhd_orszag_tang_2d_vtu.mojo` driver, which exercises:

* `ImhdGLM` with Dedner divergence cleaning,
* `DGImhdGLMSolver[2, P]` with Rusanov flux (HLLD is a one-line switch),
* `MoeRossmanith[2, P, MHD_GLM_NC]` slope limiter,
* `SSPRK3[2, P, MHD_GLM_NC]` with a CFL-driven time step, and the GLM
  damping source composed into the `compute_rhs` closure,
* `FrameWriter` + `dump_imhd_glm_frame` for VTU output,
* integrated conservation diagnostics each frame via `mesh.integrate`.

## What's modelled

Initial condition on $[0, 2\pi]^2$ with periodic BCs:

| Variable | $t = 0$ |
| -------- | ------- |
| $\rho$ | $\gamma^{2}$ |
| $u_x$ | $-\sin y$ |
| $u_y$ | $+\sin x$ |
| $B_x$ | $-\sin y$ |
| $B_y$ | $\sin(2x)$ |
| $P$ | $\gamma$ |
| $\psi$ | $0$ |

$\gamma = 5/3$, $c_h = 4$, $c_p^{2} = 0.5$. Runs to $t = 0.5$ in 11
output frames.

## Run it

```fish
pixi run example imhd_orszag_tang_2d_vtu
```

The console prints the integrated diagnostics per frame:

```text
time   ∫ρ dV   ∫E dV   ∫ψ² dV   rel Δmass   rel ΔE
0.0 109.66227112320146 173.2663883746777 0.0 0.0 0.0
0.05 109.66227112321529 173.26638837467803 5.785429809628215e-10 1.2608859452087446e-13 1.9684170505552763e-15
...
0.5 109.66227112321464 173.26638837467752 1.0308759465270991e-05 1.2012757155277556e-13 -9.842085252776382e-16
Wrote 11 frames and output/orszag_tang/orszag_tang.pvd
```

* `∫ρ dV` and `∫E dV` are conserved on the periodic domain: the relative
  drifts stay at 1e-13 and 1e-14.
* `∫ψ² dV` stays small (1e-5 at $t = 0.5$): GLM cleaning carries the
  divergence error away and damps it.

`output/orszag_tang/orszag_tang.pvd` is the time-series collection for
ParaView. `FrameWriter("orszag_tang")` creates the directory.

## Walkthrough

### Why MixedGLM

Plain ideal MHD on this problem develops a $\nabla\cdot\mathbf{B}$
error within a few time units. The error feeds back through the flux
and the simulation drifts. GLM cleaning ([Dedner et al. 2002](../numerical-methods/dedner-glm.md))
adds a scalar $\psi$ that transports the divergence error away and
damps it — the simulation runs to $t = 0.5$ with mass drift $6 \times
10^{-14}$.

### Mesh and IC

```mojo
from warpxm.prelude import *
from warpxm.physics.imhd_glm import *

comptime State = DGState[2, P, MHD_GLM_NC]

var mesh = BlockMesh[2, P](
    ctx, Real(0.0), TWO_PI, Real(0.0), TWO_PI,
    nx=NX, ny=NY, periodic_x=True, periodic_y=True,
)
var q = State(ctx, mesh.num_elements)

def initial(x: Real, y: Real) -> Array[Real, MHD_GLM_NC]:
    var rho = GAMMA * GAMMA
    var ux = -sin(y)
    var uy = sin(x)
    var bx = -sin(y)
    var by = sin(Real(2.0) * x)
    var energy = (
        GAMMA / (GAMMA - Real(1.0))
        + Real(0.5) * rho * (ux * ux + uy * uy)
        + Real(0.5) * (bx * bx + by * by)
    )
    var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
    v[IDX_RHO] = rho
    v[IDX_MX] = rho * ux
    v[IDX_MY] = rho * uy
    v[IDX_E] = energy
    v[IDX_BX] = bx
    v[IDX_BY] = by
    return v^
mesh.fill(q, initial)
```

`MHD_GLM_NC = 9`; component `IDX_PSI = 8` is the GLM scalar ψ, left at
zero. The closure returns the conserved state at a point and
`mesh.fill` evaluates it at every GLL node.

### Solver and closures

```mojo
var app = ImhdGLM(
    gas_gamma=GAMMA, min_density=Real(1e-12),
    min_pressure=Real(1e-12), cfl_max=Real(0.5),
    c_h=C_H, c_p_sq=C_P_SQ,
)
var spatial = DGImhdGLMSolver[2, P](app)
var stepper = SSPRK3[2, P, MHD_GLM_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith[2, P, MHD_GLM_NC](ctx, mesh.num_elements)
var fw = FrameWriter("orszag_tang")

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

def adjust(mut q_: State) raises {mut limiter, imm}:
    limiter.apply(ctx, mesh, q_)

def suggest_dt(mut q_: State) raises {imm} -> Real:
    return spatial.suggest_dt(ctx, mesh, q_)
```

The spatial solver handles the $\psi$-coupled flux; the parabolic GLM
damping is composed into `compute_rhs` as a source kernel that adds
$-\tfrac{c_h^{2}}{c_p^{2}}\psi$ to the $\psi$ RHS. The limiter is
NC-parameterised and works at NC = 9 without changes.

### Frame loop

```mojo
def psi_squared(v: Array[Real, MHD_GLM_NC]) -> Real:
    return v[IDX_PSI] * v[IDX_PSI]

var t: Real = 0.0
for frame in range(NUM_FRAMES):
    dump_imhd_glm_frame(fw.next_path(t), mesh, q, GAMMA)
    var mass = mesh.integrate(q, IDX_RHO)
    var energy = mesh.integrate(q, IDX_E)
    print(
        t, mass, energy, mesh.integrate_fn(q, psi_squared),
        (mass - mass_0) / mass_0, (energy - energy_0) / energy_0,
    )
    if frame < NUM_FRAMES - 1:
        t = stepper.run_until_cfl(
            ctx, q, t, T_FINAL * Real(frame + 1) / Real(NUM_FRAMES - 1),
            compute_rhs, adjust, suggest_dt, dt_interval=10,
        )

fw.write_pvd()
```

`mesh.integrate(q, comp)` and `mesh.integrate_fn(q, f)` are GLL
quadratures of a component and of any pointwise function of the node
state.

The time step is not hard-coded. `spatial.suggest_dt` takes the largest
of $|u| + \sqrt{c_s^2 + c_A^2}$ over all nodes, floors it at $c_h$, and
returns `cfl_max / ((2P + 1) · λ · (1/Δx + 1/Δy))`. `run_until_cfl`
advances to each frame time, refreshing the step every `dt_interval = 10`
steps and shortening the last step to land on the frame time exactly. It
accepts a `dt_min` argument and raises if the suggested step falls to or
below it, or is NaN, so a degenerating run fails instead of stalling.

### Athena normalisation

`examples/imhd_orszag_tang_2d_athena_vtu.mojo` is the same problem in
the Athena convention (domain $[0, 1]^2$, $\rho_0 = 25/(36\pi)$,
$p_0 = 5/(12\pi)$, $B_0 = 1/\sqrt{4\pi}$) at P = 3 on 128² cells, run
to $t = 4$ in 41 frames (about six minutes). It constructs the limiter
with `alpha_coeff=Real(100.0)`. The limiter only acts on deviations
above $\alpha = \texttt{alpha\_coeff} \cdot \Delta x^{1.5}$, and with
the default 1000 it is idle on this problem: a node then evacuates, its
sound speed grows, and the CFL step collapses (with the old fixed step
the run went NaN near $t = 0.55$). With 100 the step stays between
5.9e-5 and 9.3e-5 and $\rho \in [0.042, 0.45]$ at $t = 0.3$, against
Athena's $[0.047, 0.55]$; the limiter is somewhat more diffusive than
the reference.

## What you'll see

Open the PVD in ParaView and colour by `rho`:

* $t = 0$: smooth bipolar flow with the magnetic-field rotation
  pattern.
* $t \approx 0.2$: first shocks form at the converging-flow regions.
* $t = 0.5$: complex network of intersecting MHD shocks; characteristic
  Orszag-Tang "diamond" structure visible in $|B|^{2}$.

For comparison with the literature, the Orszag-Tang vortex at $t =
0.5$ is the standard "structure benchmark" — Brackbill, Mignone,
Stone-Gardiner all show it.

## Variations

* **HLLD flux**: change `DGImhdGLMSolver[2, P](app)` to
  `DGImhdGLMSolver[2, P, USE_HLLD=True](app)`. The shock structure
  sharpens noticeably; integrated quantities drift more slowly. The
  GLM path composes the IMHD HLLD solver with the exact 2x2
  (B_n, ψ) Riemann block — see
  [HLLD](../numerical-methods/hlld.md) — and is regression-tested
  by `tests/test_imhd_glm_decay_2d_hlld.mojo`.
* **Mesh refinement**: $\text{NX} = 128$ — the shock network is much
  cleaner, and `psi^{2}` becomes a more aggressive lower bound on the
  divergence error magnitude.
* **Removing GLM**: substitute `IMHD` for `ImhdGLM`, drop the
  `apply_glm_damping` line from `compute_rhs`, and switch
  `MHD_GLM_NC` → `MHD_NC` throughout. The simulation will diverge
  within ~0.1 time units as $\nabla\cdot\mathbf{B}$ grows.
