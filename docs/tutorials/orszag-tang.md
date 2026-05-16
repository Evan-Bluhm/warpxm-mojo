# Orszag-Tang vortex (MHD)

The Orszag-Tang vortex is *the* canonical 2D MHD benchmark — a smooth
incompressible-looking IC that develops complex shock-interaction
structure over time. This tutorial runs the full
`examples/imhd_orszag_tang_2d_vtu.mojo` driver, which exercises:

* `ImhdGLM` with Dedner divergence cleaning,
* `DGImhdGLMSolver2D` with Rusanov flux (HLLD is a one-line switch),
* `MoeRossmanith2D[P, MHD_GLM_NC]` slope limiter,
* `SSPRK3_2D[P, MHD_GLM_NC]` for time stepping, with the GLM damping
  source composed into the `compute_rhs` closure,
* `FrameWriter` + `dump_imhd_glm_2d_frame` for VTU output,
* host-side integrated conservation diagnostics each frame.

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
mkdir -p /tmp/orszag_tang
pixi run mojo run -I src examples/imhd_orszag_tang_2d_vtu.mojo
```

The console prints, per frame, the integrated diagnostics:

```text
time         ∫ρ dV            ∫E dV            ∫ψ² dV         rel Δmass   rel ΔE
-----------  ---------------- ---------------- ---------------- ----------  ----------
0.0          109.66           ...              0.0              0.0         0.0
0.05         109.66           ...              <bounded>        6e-14       8e-14
...
```

These are useful sanity checks:

* `∫ρ dV` should be exactly conserved (periodic boundaries).
* `∫E dV` should be exactly conserved.
* `∫ψ² dV` should stay bounded — the GLM cleaning dissipates the
  divergence error fast enough that ψ doesn't grow.

After the run, `/tmp/orszag_tang/orszag_tang.pvd` is the time-series
collection for ParaView.

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
var mesh = BlockMesh2D[P](
    ctx, 0.0, TWO_PI, 0.0, TWO_PI,
    nx=64, ny=64, periodic_x=True, periodic_y=True,
)
var q = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
```

`MHD_GLM_NC = 9`. The IC loop initialises ρ, momenta, energy (including
magnetic), B-components, and `IDX_PSI = 0`.

### Stepper

```mojo
var app = ImhdGLM(
    gas_gamma=Real(5.0/3.0),
    min_density=Real(1e-12),
    min_pressure=Real(1e-12),
    cfl_max=Real(0.5),
    c_h=Real(4.0),
    c_p_sq=Real(0.5),
)
var spatial = DGImhdGLMSolver2D[P](app)
var stepper = SSPRK3_2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

@parameter
def compute_rhs(
    mut q_in: DGState2D[P, MHD_GLM_NC],
    mut q_out: DGState2D[P, MHD_GLM_NC],
) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)
    apply_glm_damping_2d[P](ctx, q_in, q_out, C_H, C_P_SQ)

@parameter
def adjust(mut q_: DGState2D[P, MHD_GLM_NC]) raises:
    limiter.apply(ctx, mesh, q_)
```

The spatial solver handles the $\psi$-coupled flux; the parabolic GLM
damping is composed into `compute_rhs` as a source kernel that adds
$-\tfrac{c_h^{2}}{c_p^{2}}\psi$ to the $\psi$ RHS. The limiter is
NC-parameterised and works at NC=9 without changes.

### Inside the time loop

```mojo
for frame in range(NUM_FRAMES):
    dump_imhd_glm_2d_frame[P, MHD_GLM_NC](fw.next_path(t), ctx, mesh, q, GAMMA)
    mass = _integrate_field[IDX_RHO](ctx, q, mesh)
    e_tot = _integrate_field[IDX_E](ctx, q, mesh)
    psi2 = _integrate_squared_field[IDX_PSI](ctx, q, mesh)
    # print diagnostics
    if frame < NUM_FRAMES - 1:
        # advance to next frame time
        ...
```

The `_integrate_field` / `_integrate_squared_field` helpers in the
driver are short host-side reductions using GLL quadrature weights —
the exact-mean integration property of the basis means these come
out conservative to machine precision.

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

* **HLLD flux**: change `DGImhdGLMSolver2D[P](app)` to
  `DGImhdGLMSolver2D[P, USE_HLLD=True](app)`. The shock structure
  sharpens noticeably; integrated quantities drift more slowly.
  (Note: HLLD wiring for the GLM solver is pending — currently it's
  only on the plain `DGImhdSolver2D`. Track this in
  [Roadmap](../contributing/index.md#roadmap).)
* **Mesh refinement**: $\text{NX} = 128$ — the shock network is much
  cleaner, and `psi^{2}` becomes a more aggressive lower bound on the
  divergence error magnitude.
* **Removing GLM**: substitute `IMHD` for `ImhdGLM`, drop the
  `apply_glm_damping_2d` line from `compute_rhs`, and switch
  `MHD_GLM_NC` → `MHD_NC` throughout. The simulation will diverge
  within ~0.1 time units as $\nabla\cdot\mathbf{B}$ grows.
