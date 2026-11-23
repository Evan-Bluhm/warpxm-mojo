# `warpxm.variable`

Conserved-variable storage: `DGState[NDIM, P, NC]` for configuration-space
fields and `PhaseState1D1V` / `PhaseState1D2V` for the kinetic
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
| `DGState[NDIM, P, NC].num_nodes_per_elem` | comptime, `(P+1)^NDIM` |
| `DGState[NDIM, P, NC].index(elem, node, comp)` | static; flat buffer index |

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
`PhaseState.index(elem, node, comp)` gives the flat index. For `XDIM = 1`
the layout is identical to the two original types below.

```mojo
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.phase_state_1d2v import PhaseState1D2V

var f = PhaseState1D1V[P_x, P_v, VLASOV_NC](ctx, num_x, num_v)
var g = PhaseState1D2V[P_x, P_v1, P_v2, VLASOV_NC](ctx, num_x, num_v1, num_v2)
```

Layout, with the x node index fastest inside a phase element:

```text
1D1V: buf[((phase_e * (P_v+1) + iv) * (P_x+1) + ix) * NC + c]
      phase_e = ex * num_v + ev
1D2V: buf[(phase_e * NPE + (iv2 * (P_v1+1) + iv1) * (P_x+1) + ix) * NC + c]
      phase_e = (ex * num_v1 + ev1) * num_v2 + ev2
      NPE = (P_x+1) * (P_v1+1) * (P_v2+1)
```

Members: `buf`, `buf_len`, `total_nodes`, `num_phase_elements`, `num_x`,
`num_v` (or `num_v1`, `num_v2`). The phase meshes provide `fill` and
`visit_nodes` for these states; there is no `index` helper. See
[Vlasov-Maxwell](../physics/vlasov.md).
