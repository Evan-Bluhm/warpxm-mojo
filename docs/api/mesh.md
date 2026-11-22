# `warpxm.mesh`

Uniform block meshes in 1D, 2D, and 3D, and the 1D1V / 1D2V phase-space
meshes. Each carries face / element connectivity, per-face
boundary-condition tags, and per-node coordinates as device buffers, plus
host-side helpers for initial conditions and diagnostics.

## `BlockMesh[NDIM, P]`

One struct serves every dimension. It has one constructor per dimension,
selected by the arguments:

```mojo
from warpxm.mesh.block import (
    BlockMesh,
    GHOST_NEG, GHOST_POS,
    FACE_DIR_LEFT, FACE_DIR_RIGHT, FACE_DIR_BOTTOM, FACE_DIR_TOP,
    FACE_DIR_DOWN, FACE_DIR_UP,
)

var mesh1 = BlockMesh[1, P](
    ctx, x_min, x_max, num_elements,
    periodic=True,
    bc_left=BC_COPYOUT,
    bc_right=BC_COPYOUT,
)

var mesh2 = BlockMesh[2, P](
    ctx, x_min, x_max, y_min, y_max,
    nx=NX, ny=NY,
    periodic_x=True, periodic_y=True,
    bc_left=BC_COPYOUT,
    bc_right=BC_COPYOUT,
    bc_bottom=BC_WALL_FREESLIP,
    bc_top=BC_WALL_FREESLIP,
)

var mesh3 = BlockMesh[3, P](
    ctx, x_min, x_max, y_min, y_max, z_min, z_max,
    nx=NX, ny=NY, nz=NZ,
    periodic_x=False, periodic_y=False, periodic_z=False,   # default True
    bc_left=BC_COPYOUT,   bc_right=BC_COPYOUT,    # x axis
    bc_bottom=BC_COPYOUT, bc_top=BC_COPYOUT,      # y axis
    bc_down=BC_COPYOUT,   bc_up=BC_COPYOUT,       # z axis
)
```

Using a constructor on the wrong `NDIM` is a compile error. The BC
parameters for any axis with `periodic_<axis>=True` are ignored.

Fields exposed (axes beyond `NDIM` have one element and drop out of
every sum and product):

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `mesh.nx`, `ny`, `nz` | `Int` | elements along each axis |
| `mesh.num_elements` | `Int` | `nx * ny * nz` |
| `mesh.num_faces` | `Int` | total faces; per axis, `n` along a periodic axis and `n + 1` otherwise |
| `mesh.x_min`, `x_max`, `y_min`, … `z_max` | `Real` | domain extent |
| `mesh.dx`, `dy`, `dz` | `Real` | uniform element widths |
| `mesh.jacobian_x`, `jacobian_y`, `jacobian_z` | `Real` | `dx / 2` etc. |
| `mesh.periodic_x`, `periodic_y`, `periodic_z` | `Bool` | per-axis periodicity |
| `mesh.node_coords_buf` | `DeviceBuffer[Real.dtype]` | node coordinates, `[(e * nodes_per_elem + n) * NDIM + d]` |
| `mesh.face_neg_elem_buf`, `face_pos_elem_buf` | `DeviceBuffer[INDEX_DTYPE]` | element on the negative / positive side of every face; `GHOST_NEG` / `GHOST_POS` on a boundary |
| `mesh.elem_face_dir_buf` | `DeviceBuffer[INDEX_DTYPE]` | face id on each side of every element, `[e * 2 * NDIM + dir]` |
| `mesh.face_bc_buf` | `DeviceBuffer[INDEX_DTYPE]` | BC kind per face (`BC_INTERIOR` on interior faces) |

Numbering:

* element `e = (k * ny + j) * nx + i`;
* local node `n = (kn * (P + 1) + jn) * (P + 1) + in`;
* side `dir = 2 * axis` for the negative side and `2 * axis + 1` for the
  positive side, named `FACE_DIR_LEFT`, `RIGHT`, `BOTTOM`, `TOP`, `DOWN`,
  `UP`;
* faces are grouped by axis (all x-normal faces first).

Geometry helpers: `mesh.volume()`, `mesh.inv_dx_sum()` and
`mesh.min_dx()` (smallest element width).

## Periodic vs non-periodic

`periodic_*=True` configures the face buffers so faces wrap around —
i.e. the right-most face's `face_pos_elem` references element 0, and
vice versa. In that mode the per-side `bc_*` parameters are ignored.

`periodic_*=False` adds an extra row/column of "ghost" faces on the
boundary. The per-axis `bc_*` parameters select what happens at those
ghost faces (see [`core.bc`](core.md)).

## Host-side helpers

`BlockMesh` owns the node ordering and the `DGState` buffer
layout, so a driver or test never indexes the flat buffer. The closures
take `(x)`, `(x, y)` or `(x, y, z)` according to the mesh dimension
(`fill`, `visit_nodes` and `l2_error` have one overload per dimension);
the 2D forms are shown.

| Method | Result |
| ------ | ------ |
| `mesh.fill(q, ic)` | sets every node of `q` to `ic(x, y) -> Array[Real, NC]` |
| `mesh.visit_nodes(q, visit)` | calls `visit(x, y, w, q_node)` at every node |
| `mesh.integrate(q, comp)` | $\int q_\text{comp}\,dV$ by GLL quadrature |
| `mesh.integrate_fn(q, f)` | $\int f(q)\,dV$ for `f(q_node) -> Real` |
| `mesh.l2_error(q, comp, exact)` | $\sqrt{\int (q_\text{comp} - \text{exact}(x, y))^2\,dV}$ |
| `mesh.volume()` | measure of the domain |
| `mesh.inv_dx_sum()` | $\sum_d 1/\Delta x_d$, the mesh factor in the CFL bound |
| `mesh.min_dx()` | smallest element width |

```mojo
def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
    var rho = Real(1.0) + Real(0.2) * sin(TWO_PI * x)
    return [rho, rho, Real(0.0), Real(0.0), Real(2.5) + Real(0.5) * rho]
mesh.fill(q, initial)

var mass = mesh.integrate(q, IDX_RHO)

def kinetic(v: Array[Real, EULER_NC]) -> Real:
    return Real(0.5) * v[IDX_MX] * v[IDX_MX] / v[IDX_RHO]
var ke = mesh.integrate_fn(q, kinetic)

def exact_rho(x: Real, y: Real) {imm} -> Real:
    return Real(1.0) + Real(0.2) * sin(TWO_PI * (x - t))
var err = mesh.l2_error(q, IDX_RHO, exact_rho)
```

`visit_nodes` is the general form. `w` is the node's quadrature weight
(GLL weight times the element Jacobian), so summing `w * g` over the
visits gives $\int g\,dV$. Accumulate through `mut` captures:

```mojo
var rho_min: Real = 1.0e300
var l1: Real = 0.0
def visit(x: Real, y: Real, w: Real, v: Array[Real, EULER_NC]) {mut rho_min, mut l1}:
    rho_min = fmin(rho_min, v[IDX_RHO])
    l1 += w * fabs(v[IDX_RHO] - Real(1.0))
mesh.visit_nodes(q, visit)
```

Closure capture lists follow the usual rule: a closure that reads an
outer runtime variable needs one (`{imm}`, or `{mut name}` for what it
writes); one that uses only its arguments and `comptime` constants needs
none.

Notes:

* The quadrature is the collocated GLL rule, exact for polynomials of
  degree $2P - 1$ per direction. `l2_error` therefore measures the nodal
  error with that rule; it does not over-integrate.
* All of these copy the whole state to the host (and `fill` copies it
  back). Use them for setup and diagnostics, not inside the time loop of
  a large run.
* For a sample at one specific node use `DGState.index` (see
  [`variable`](variable.md)).

`tests/test_mesh_helpers.mojo` checks every helper against the mesh's
own coordinate buffer and closed-form integrals.

## Phase-space meshes

```mojo
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V

var m1 = PhaseMesh1D1V[P_x, P_v](
    ctx, x_min, x_max, v_min, v_max,
    num_x=NX, num_v=NV, periodic_x=True,
)
var m2 = PhaseMesh1D2V[P_x, P_v1, P_v2](
    ctx, x_min, x_max, v1_min, v1_max, v2_min, v2_max,
    num_x=NX, num_v1=NV1, num_v2=NV2, periodic_x=True,
)
```

Tensor-product meshes over $(x, v)$ or $(x, v_1, v_2)$ with independent
polynomial orders per axis. The velocity axes are never periodic: their
boundary faces carry `BC_VLASOV_OUTFLOW`. `bc_left` / `bc_right` apply to
the x boundaries when `periodic_x=False`.

Helpers act on component 0 of a `PhaseState` (the distribution function):

| Method | Closure |
| ------ | ------- |
| `mesh.fill(f, ic)` | `ic(x, v) -> Real` or `ic(x, v1, v2) -> Real` |
| `mesh.visit_nodes(f, visit)` | `visit(x, v, w, f_node)` or `visit(x, v1, v2, w, f_node)`; `w` is the phase-space quadrature weight |

See [Vlasov-Maxwell](../physics/vlasov.md).

## Mesh stores connectivity, not state

The mesh never holds physics state: that is `variable.DGState[NDIM, P, NC]`.
A mesh is reused across solves; states are reset per problem.
