# Nodal DG basics

This page sketches just enough discontinuous-Galerkin theory to make the
code readable. It's not a textbook substitute — see Hesthaven & Warburton
*Nodal Discontinuous Galerkin Methods* (Springer 2008) for the full
treatment.

## The strong form

For a conservation law

$$\partial_t \mathbf{q} + \nabla\cdot\mathbf{F}(\mathbf{q}) = 0$$

we partition the domain into elements. On each element we represent
each component of $\mathbf{q}$ as a tensor-product polynomial of degree
$P$ — meaning $(P+1)^d$ nodal values for a $d$-dimensional element.

The **strong form** DG residual on element $K$ is

$$\partial_t \mathbf{q}_h + \nabla\cdot\mathbf{F}_h
  + \mathcal{L}^{-1}\!\left[(\mathbf{F}^\ast - \mathbf{F}_h)\cdot\hat{n}\right] = 0,$$

where $\mathcal{L}$ is the GLL mass matrix on the element, and
$\mathbf{F}^\ast$ is the *numerical flux* on the element interfaces. The
two pieces are computed separately in our DG kernels:

* the **volume term** $\nabla\cdot\mathbf{F}_h$ comes from differentiating
  the polynomial representation at GLL nodes,
* the **surface term** $\mathcal{L}^{-1}\,(\mathbf{F}^\ast -
  \mathbf{F}_h)\cdot\hat{n}$ is applied only at face GLL nodes — for
  GLL bases this localisation is exact.

In the code those two contributions appear as `dfdx` (volume) and
`surface` (face) arrays in every `_*_rhs_kernel_*` function.

## Gauss-Lobatto-Legendre nodes

We use **Gauss-Lobatto-Legendre (GLL) nodes** as the polynomial basis.
GLL nodes include the endpoints of the reference interval $[-1, 1]$,
which means:

* face values are *exactly* the values stored at the boundary nodes
  (no extrapolation needed),
* the basis is *nodal* — coefficients equal physical values at the nodes,
* GLL quadrature with `P + 1` points integrates polynomials of degree
  $2P - 1$ exactly.

The reference-node coordinates and weights for $P = 1, 2, 3, 4$ are
tabulated in [`src/warpxm/basis/nodal_basis_1d.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/src/warpxm/basis/nodal_basis_1d.mojo).

Convergence at smooth problems goes as $h^{P+1}$. The 1D advection
convergence test verifies this:

| Polynomial order $P$ | Observed rate |
| -------------------: | ------------- |
| 1 | 1.98 |
| 2 | 3.01 |
| 3 | 3.99 |

## Tensor-product elements

In 2D and 3D we don't use unstructured triangles or tetrahedra. Every
element is a tensor-product of the 1D basis: an $(P+1)^2$ quadrilateral
in 2D, an $(P+1)^3$ hexahedron in 3D. The volume and surface terms then
factor naturally — the volume derivative along axis $\hat{x}$ is the 1D
derivative matrix applied along one tensor axis, leaving the others
untouched.

Concretely, the 2D Euler kernel computes

```mojo
var dfdx = ...   # ∂F/∂x at this node, by 1D differentiation along i
var dgdy = ...   # ∂G/∂y at this node, by 1D differentiation along j
```

and the surface term is non-zero only at the four faces.

## Numerical fluxes

At element interfaces we need a single *upwind* flux. warpxm-mojo
implements two:

* **Rusanov / Lax-Friedrichs** — the default, simple, dissipative. Picks
  the maximum-speed wave to set the diffusion: $\mathbf{F}^\ast = \tfrac{1}{2}(\mathbf{F}_L
  + \mathbf{F}_R) - \tfrac{1}{2}\lambda_{\max}\,(\mathbf{q}_R - \mathbf{q}_L)$.
* **HLLD** (MHD only) — five-state Miyoshi-Kusano solver that resolves
  fast magnetosonic, Alfvén and contact waves. Much sharper shocks.

See [Numerical Methods](../numerical-methods/index.md) for the gory
details.

## Boundary conditions

Periodic, copyout (zero-gradient), no-slip wall, perfectly-conducting
(PEC) and perfectly-magnetic (PMC) wall conditions are all implemented
by setting an appropriate **ghost-state** at the boundary and re-using
the same numerical-flux machinery. The BC is selected per-axis in the
mesh constructor:

```mojo
var mesh = BlockMesh2D[P](
    ctx, x_min, x_max, y_min, y_max,
    nx=NX, ny=NY,
    periodic_x=False, periodic_y=True,
    bc_left=BC_WALL_FREESLIP, bc_right=BC_COPYOUT,
)
```

See the [`core.bc`](../api/core.md) module for the full enum and
[apps](../api/apps.md) for per-physics BC behaviour.

## Where this lives in the code

| Concept                       | File                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| GLL nodes + weights           | `src/warpxm/basis/nodal_basis_1d.mojo`                        |
| Mesh + ghost wiring           | `src/warpxm/mesh/block_{1,2,3}d.mojo`                         |
| Shared DG core (per dim)      | `src/warpxm/spatial/dg_core_{1,2,3}d.mojo`                    |
| Advection DG kernel           | `src/warpxm/spatial/dg_{1,2,3}d.mojo` (the un-prefixed files) |
| Per-physics DG kernels        | `src/warpxm/spatial/dg_{euler,maxwell,imhd,imhd_glm}_{N}d.mojo` |
| Volume vs surface split       | Each kernel — search for `dfdx` (volume) and `surface` (face) |

Reading just one DG kernel — `spatial/dg_euler_2d.mojo` is a good
target — is usually enough to internalise the pattern.
