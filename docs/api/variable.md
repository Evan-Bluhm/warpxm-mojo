# `warpxm.variable`

Conserved-variable storage. `DGState1D / 2D / 3D` are thin wrappers
around a flat `DeviceBuffer[Real.dtype]` with the right indexing
helpers and size accounting.

## `DGStateNd[P, NC]`

```mojo
from warpxm.variable.state_1d import DGState1D
from warpxm.variable.state_2d import DGState2D
from warpxm.variable.state_3d import DGState3D

var q = DGState1D[P, NC](ctx, num_elements)
var q = DGState2D[P, NC](ctx, num_elements)
var q = DGState3D[P, NC](ctx, num_elements)
```

Fields:

| Field | Meaning |
| ----- | ------- |
| `q.buf` | `DeviceBuffer[Real.dtype]` — flat storage |
| `q.buf_len` | total scalar count (`num_elements * (P+1)^d * NC`) |
| `q.total_nodes` | total node count (no NC factor — useful for scalar IC loops) |
| `q.num_elements` | element count, copied from the mesh |

## Memory layout

The conventional flat index is:

```text
q.buf[ (e * nodes_per_elem + n) * NC + c ]
```

* `e` — element index, `0 ≤ e < num_elements`
* `n` — within-element node index, `0 ≤ n < (P+1)^d`
  * In 2D: `n = jn * (P+1) + in_`
  * In 3D: `n = kn * (P+1)*(P+1) + jn * (P+1) + in_`
* `c` — component index, `0 ≤ c < NC` (e.g., `IDX_RHO`, `IDX_MX`, …)

This layout is used by every DG kernel, limiter, positivity preserver,
and VTU writer. Don't reorder it.

## IC pattern

```mojo
var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
for j in range(NY):
    for i in range(NX):
        var xl = mesh.x_min + Real(i) * mesh.dx
        var e = j * NX + i
        for jn in range(P1):
            for in_ in range(P1):
                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                # ... compute physical state here ...
                var n = jn * P1 + in_
                var b = (e * nodes_per_elem + n) * NC
                host[b + IDX_RHO] = rho
                host[b + IDX_MX] = rho * u
                ...
ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
```

Every example in `examples/` follows this pattern. The IC loop runs on
the host, building a host buffer, then `enqueue_copy` uploads it.

## Reading back to host

```mojo
with q.buf.map_to_host() as h:
    for i in range(q.buf_len):
        print(h[i])
```

`map_to_host()` is a context manager that handles synchronisation. The
returned host view is read-write — if you modify it, the changes are
written back when the `with` block exits.

## Multi-state problems

The plasma stepper bundles three states (two fluids + Maxwell). You
construct each one separately:

```mojo
var q_s1 = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
var q_s2 = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
var q_maxwell = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)
```

The plasma stepper threads them through coupled RK stages.
