# `warpxm.basis`

Gauss-Lobatto-Legendre (GLL) basis tables. The only file is
[`basis/nodal_basis_1d.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/src/warpxm/basis/nodal_basis_1d.mojo).
Tensor-product 2D and 3D bases are derived from the 1D basis on the
fly.

## `NodalBasis1D[P]`

```mojo
from warpxm.basis.nodal_basis_1d import NodalBasis1D

comptime P = 2
var nodes = NodalBasis1D[P].nodes()                # InlineArray[Real, P+1]
var weights = NodalBasis1D[P].weights()            # InlineArray[Real, P+1]
var D_flat = NodalBasis1D[P].diff_matrix_flat()    # InlineArray[Real, (P+1)*(P+1)]
                                                   # row-major D[i,j] = l_j'(ξ_i)
var i_left  = NodalBasis1D[P].left_face_node()     # = 0
var i_right = NodalBasis1D[P].right_face_node()    # = P
var num_nodes = NodalBasis1D[P].num_nodes          # = P + 1
```

## Supported orders

| `P` | Nodes / element | Theoretical convergence rate | Observed (1D advection) |
| --- | -------------- | ---------------------------- | ----------------------- |
| 1 | 2  | $h^{2}$ | 1.98 |
| 2 | 3  | $h^{3}$ | 3.01 |
| 3 | 4  | $h^{4}$ | 3.99 |
| 4 | 5  | $h^{5}$ | floored by SSPRK3 (3rd-order in time) |

The $P = 4$ rate is not testable with SSPRK3 — the temporal error
dominates. Use Dormand-Prince DP45 to see fifth-order spatial
convergence at $P = 4$.

## Reference interval

Nodes are stored on $[-1, 1]$. Physical-space conversion happens in the
mesh constructor:

```mojo
x_physical = mesh.x_min + Real(e) * mesh.dx + (Real(1.0) + xi) * mesh.jacobian
```

with `mesh.jacobian = mesh.dx / 2` and `xi` a reference-node coordinate
from `NodalBasis1D[P].nodes()`. Every example demonstrates this
conversion in its IC loop.

## Why GLL specifically

GLL nodes include the reference-interval endpoints $\xi = \pm 1$, so
face values are directly the boundary-node values (no extrapolation).
This is what makes the strong-form DG kernel work without ever solving
a small mass system per element. See
[Nodal DG basics](../concepts/nodal-dg.md) for the underlying
property.
