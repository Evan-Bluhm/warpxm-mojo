# `warpxm.variable`

Conserved-variable storage: `DGState[NDIM, P, NC]` for configuration-space
fields and `PhaseState[XDIM, VDIM, P_x, P_v, NC]` for the kinetic
distribution function. Each is a thin wrapper around one flat
`DeviceBuffer[Real.dtype]`.

## `DGState[NDIM, P, NC]`

```mojo
from warpxm.variable.state import DGState

var q1 = DGState[1, P, NC](ctx, num_elements)
var q2 = DGState[2, P, NC](ctx, mesh.num_elements)
var q3 = DGState[3, P, NC](ctx, mesh.num_elements)
```

One struct serves every dimension: only the node count per element,
`(P+1)^NDIM`, depends on `NDIM`. That is why the time steppers and the
pointwise source kernels are written once. The buffer is zero-filled on
construction.

| Member | Meaning |
| ------ | ------- |
| `q.buf` | `DeviceBuffer[Real.dtype]`, flat storage |
| `q.buf_len` | total scalar count, `total_nodes * NC` |
| `q.total_nodes` | `num_elements * (P+1)^NDIM` |
| `q.num_elements` | element count |
| `q.num_ghost` | ghost elements stored after the owned ones (0 on one rank) |
| `q.alloc_len` | length of `q.buf` including the ghosts |
| `DGState[NDIM, P, NC].num_nodes_per_elem` | comptime, `(P+1)^NDIM` |
| `DGState[NDIM, P, NC].index(elem, node, comp)` | static; flat buffer index |

A driver that should run on several ranks passes the mesh's ghost count
as well, `DGState[2, P, NC](ctx, mesh.num_elements, mesh.num_ghost)`.
`num_elements`, `total_nodes` and `buf_len` always count owned data, so
kernels sized by them never touch a ghost
([Running on several GPUs](../concepts/parallel.md)).

A driver usually names the type once:

```mojo
comptime State = DGState[2, P, EULER_NC]
```

## Memory layout

```text
q.buf[(e * (P+1)^NDIM + n) * NC + c]
```

* `e`: element index, `0 ≤ e < num_elements` (row-major over the mesh,
  x fastest: `e = (k * ny + j) * nx + i`)
* `n`: node index within the element, x fastest:
  `n = (kn * (P+1) + jn) * (P+1) + in`
* `c`: component index, `0 ≤ c < NC` (`IDX_RHO`, `IDX_MX`, …)

Every DG kernel, limiter, positivity preserver and the VTU writer use
this layout. Host code should not re-derive it: use the mesh helpers, or
`index` when a specific node is needed.

## Setting and reading a state

Initial conditions and diagnostics go through the mesh (see
[`mesh`](mesh.md#host-side-helpers)):

```mojo
def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
    return [Real(1.0), Real(0.0), Real(0.0), Real(0.0), Real(2.5)]
mesh.fill(q, initial)
var mass = mesh.integrate(q, IDX_RHO)
```

To sample particular nodes, map the buffer and use `index`:

```mojo
var mid = State.num_nodes_per_elem // 2
with q.buf.map_to_host() as h:
    var rho_centre = h[State.index(elem, mid, IDX_RHO)]
```

`map_to_host()` is a context manager that synchronises the device queue
and copies the buffer to the host. The view is read-write: changes are
written back when the `with` block exits. It copies the whole buffer, so
keep it out of the time loop of a large run.

## Multi-state problems

The coupled steppers take several states, each constructed separately:

```mojo
var q_s1 = DGState[2, P, EULER_NC](ctx, mesh.num_elements)
var q_s2 = DGState[2, P, EULER_NC](ctx, mesh.num_elements)
var q_maxwell = DGState[2, P, MAXWELL_NC](ctx, mesh.num_elements)
```

## Phase-space states

```mojo
from warpxm.variable.phase_state import PhaseState

var f = PhaseState[1, 3, P_x, P_v, VLASOV_NC](ctx, mesh.num_phase_elements)
```

`PhaseState[XDIM, VDIM, P_x, P_v, NC]` is the generic phase-space state.
Phase elements are numbered with the last axis innermost, local nodes
with the first axis innermost, and
`PhaseState.index(elem, node, comp)` gives the flat index:

```text
phase element  e = ((e_0 * n_1 + e_1) * n_2 + ...) + e_last     (axes: x..., then v...)
local node     n = i_0 + s_1 * i_1 + s_2 * i_2 + ...            (s_a: node count of the axes before a)
buffer         buf[(e * nodes_per_elem + n) * NC + c]
```

Members: `buf`, `buf_len`, `total_nodes`, `num_phase_elements`. The
phase mesh provides `fill`, `visit_nodes` and `integrate` for these
states. See [Vlasov-Maxwell](../physics/vlasov.md).
