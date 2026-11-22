# Ideal MHD

Ideal magnetohydrodynamics in conservative form with a γ-law fluid and
$\mu_0 = 1$:

$$
\partial_t
\begin{pmatrix}\rho \\ \rho\mathbf{u} \\ E \\ \mathbf{B}\end{pmatrix}
+
\nabla\cdot
\begin{pmatrix}
\rho\mathbf{u} \\
\rho\mathbf{u}\mathbf{u} + (P + \tfrac{1}{2}|\mathbf{B}|^{2})\,I - \mathbf{B}\mathbf{B} \\
(E + P + \tfrac{1}{2}|\mathbf{B}|^{2})\mathbf{u} - (\mathbf{u}\cdot\mathbf{B})\mathbf{B} \\
\mathbf{u}\mathbf{B} - \mathbf{B}\mathbf{u}
\end{pmatrix}
= \mathbf{0}.
$$

Total energy includes magnetic contribution:

$$E = \frac{P}{\gamma - 1} + \tfrac{1}{2}\rho|\mathbf{u}|^{2} + \tfrac{1}{2}|\mathbf{B}|^{2}.$$

## Module

```mojo
from warpxm.apps.imhd import IMHD
```

```mojo
var app = IMHD(
    gas_gamma=Real(5.0 / 3.0),
    min_density=Real(1e-12),
    min_pressure=Real(1e-12),
    cfl_max=Real(0.5),
)
```

## State layout

```mojo
from warpxm.apps.imhd import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
# MHD_NC = 8
```

| Component | Index | Meaning |
| --------- | ----- | ------- |
| `q[IDX_RHO]` | 0 | Density $\rho$ |
| `q[IDX_MX]`  | 1 | $\rho u_x$ |
| `q[IDX_MY]`  | 2 | $\rho u_y$ |
| `q[IDX_MZ]`  | 3 | $\rho u_z$ |
| `q[IDX_E]`   | 4 | Total energy (incl. $\tfrac{1}{2}|\mathbf{B}|^{2}$) |
| `q[IDX_BX]`  | 5 | $B_x$ |
| `q[IDX_BY]`  | 6 | $B_y$ |
| `q[IDX_BZ]`  | 7 | $B_z$ |

## Boundary conditions

The ideal-MHD solvers (2D and 3D) dispatch on each `bc_kind`:

| BC | Behaviour |
| -- | --------- |
| Periodic | Implicit. |
| `BC_COPYOUT` | Zero-gradient outflow. |
| `BC_WALL_FREESLIP` | Reflective wall (normal momentum flipped). |
| `BC_PEC_WALL` | Perfectly-conducting wall (normal momentum and normal B flipped). |

`tests/test_imhd_walls.mojo` runs a closed box with each wall type: both
conserve mass, and the PEC wall also conserves total energy.

**GLM-MHD caveat:** `DGImhdGLMSolver` treats every non-periodic
`bc_kind` as copy-out. Use them on periodic or outflow domains.

## Time step

`spatial.suggest_dt(ctx, mesh, q)` uses
$\lambda = \max(|\mathbf{u}| + \sqrt{c_s^2 + c_A^2})$, the fast speed
perpendicular to $\mathbf{B}$, which bounds the fast speed in every
direction. For GLM-MHD the bound is floored at $c_h$. See
[`spatial`](../api/spatial.md#cfl-time-step).

## Numerical flux

Two solvers, selectable at compile time:

* **Rusanov / Lax-Friedrichs** (default). Robust, dissipative.
  $\lambda_{\max} = |u_n| + c_f$ where $c_f$ is the fast magnetosonic
  speed.
* **HLLD** — five-state Miyoshi-Kusano (2005) solver. Resolves the
  fast, Alfvén, and contact waves. Much sharper shocks.

```mojo
var spatial = DGImhdSolver[2, P, USE_HLLD=True](app)
```

See [Numerical Methods → HLLD](../numerical-methods/hlld.md) for the
algorithm.

## Divergence cleaning (Dedner GLM)

Ideal MHD as discretised is *not* divergence-free in the cell-mean
sense; small $\nabla\cdot\mathbf{B}$ errors leak in at shocks. The
**Dedner GLM (mixed-form generalised Lagrange multiplier)** scheme
augments the system with a scalar $\psi$ that transports and damps
$\nabla\cdot\mathbf{B}$ at a chosen speed $c_h$:

$$\partial_t\psi + c_h^{2}\,\nabla\cdot\mathbf{B} = -\tfrac{c_h^{2}}{c_p^{2}}\,\psi.$$

```mojo
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC, IDX_PSI
# MHD_GLM_NC = 9 — adds q[IDX_PSI] at the end

var app = ImhdGLM(
    gas_gamma=Real(5.0/3.0),
    min_density=Real(1e-12),
    min_pressure=Real(1e-12),
    cfl_max=Real(0.5),
    c_h=Real(4.0),
    c_p_sq=Real(0.5),
)
```

See [Numerical Methods → Dedner GLM](../numerical-methods/dedner-glm.md).

## Limiter and positivity

Use the NC-parameterised Moe-Rossmanith limiter at NC=8 (ideal MHD) or
NC=9 (GLM). For strong shocks add the Zhang-Shu MHD positivity
preserver — its pressure step is a cubic in θ because $|B|^{2}$
contributes; the implementation bisects, and only at nodes that violate
the bound.

```mojo
from warpxm.prelude import *
from warpxm.physics.imhd import *

var stepper = SSPRK3[2, P, MHD_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith[2, P, MHD_NC](ctx, mesh.num_elements)
var positivity = PositivityMHD[2, P, MHD_NC](mesh.num_elements, gamma=GAMMA)

def adjust(mut q_: DGState[2, P, MHD_NC]) raises {mut limiter, mut positivity, imm}:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)
```

The limiter only acts on deviations above
$\alpha = \texttt{alpha\_coeff}\cdot\Delta x^{1.5}$ (default
coefficient 1000), which on coarse meshes can exceed the solution scale
and leave the limiter idle; see
[Moe-Rossmanith](../numerical-methods/moe-rossmanith.md).

## Examples and tests

| File | Description |
| ---- | ----------- |
| [`examples/imhd_orszag_tang_2d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/imhd_orszag_tang_2d_vtu.mojo) | Orszag-Tang vortex, GLM cleaning, VTU output. Walkthrough in [Tutorials](../tutorials/orszag-tang.md). |
| [`examples/imhd_briowu_2d_hlld_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/imhd_briowu_2d_hlld_vtu.mojo) | 2D Brio-Wu with HLLD flux. |
| [`examples/imhd_strong_blast_2d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/imhd_strong_blast_2d_vtu.mojo) | Strong MHD blast exercising positivity preserver. |
| [`examples/imhd_blast_3d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/imhd_blast_3d_vtu.mojo) | 3D MHD blast. |

Notable test results:

* **Brio-Wu 2D + 3D** — y-drift 2.8 × 10⁻¹³ (2D); both y- and z-drift
  exactly 0 in 3D thanks to permutation-symmetric extrusion.
* **Alfvén 3D** — B_y / B_z L2 2.4 × 10⁻⁵ bit-identical.
* **Orszag-Tang** — mass drift 6 × 10⁻¹⁴, ψ² bounded.
* **3D MHD blast** — $\int B_x\,dV$ preserved to 4 × 10⁻¹⁵
  (Faraday's law on periodic mesh).
* **Strong MHD blast (2D and 3D, $\Delta P = 100$ on $P_{\text{amb}} = 0.1$, a ratio of 1000)**
  — `min P_gas` clamped to `p_eps = 1e-10` by the positivity preserver;
  mass and $\int B_x$ conserved to 2 × 10⁻¹³.
