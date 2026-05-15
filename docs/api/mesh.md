# `warpxm.mesh`

Uniform block meshes in 1D, 2D, and 3D. Each carries face / element
connectivity, per-face boundary-condition tags, and per-node physical-
space coordinates as device buffers.

## `BlockMesh1D[P]`

```mojo
from warpxm.mesh.block_1d import BlockMesh1D, GHOST_LEFT, GHOST_RIGHT

var mesh = BlockMesh1D[P](
    ctx, x_min, x_max, num_elements,
    periodic=True,
    bc_left=BC_COPYOUT,
    bc_right=BC_COPYOUT,
)
```

Fields exposed:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `mesh.num_elements` | `Int` | number of cells |
| `mesh.x_min`, `mesh.x_max` | `Real` | domain extent |
| `mesh.dx` | `Real` | uniform cell width |
| `mesh.jacobian` | `Real` | `dx / 2` |
| `mesh.num_faces` | `Int` | `num_elements + 1` (non-periodic) or `num_elements` (periodic) |
| `mesh.node_x_buf` | `DeviceBuffer[Real.dtype]` | physical x at each GLL node |
| `mesh.face_left_buf`, `face_right_buf` | `DeviceBuffer[INDEX_DTYPE]` | element index on each side of every face; ghost-sentinel for boundary faces |
| `mesh.elem_left_face_buf`, `elem_right_face_buf` | `DeviceBuffer[INDEX_DTYPE]` | face id on each side of every element |

## `BlockMesh2D[P]`

```mojo
from warpxm.mesh.block_2d import (
    BlockMesh2D,
    GHOST_NEG, GHOST_POS,
    FACE_DIR_LEFT, FACE_DIR_RIGHT, FACE_DIR_BOTTOM, FACE_DIR_TOP,
)

var mesh = BlockMesh2D[P](
    ctx, x_min, x_max, y_min, y_max,
    nx=NX, ny=NY,
    periodic_x=True, periodic_y=True,
    bc_left=BC_COPYOUT,
    bc_right=BC_COPYOUT,
    bc_bottom=BC_WALL_FREESLIP,
    bc_top=BC_WALL_FREESLIP,
)
```

Adds `dy`, `jacobian_x = dx/2`, `jacobian_y = dy/2`. Faces are
indexed in 4 directions: `FACE_DIR_LEFT/RIGHT/BOTTOM/TOP`. Boundary
condition selection is per-axis. The `node_xy_buf` device buffer
carries x and y physical coordinates at every GLL node.

## `BlockMesh3D[P]`

```mojo
from warpxm.mesh.block_3d import (
    BlockMesh3D,
    GHOST_NEG_3D, GHOST_POS_3D,
    FACE_DIR_LEFT_3D, FACE_DIR_RIGHT_3D,
    FACE_DIR_BOTTOM_3D, FACE_DIR_TOP_3D,
    FACE_DIR_DOWN_3D, FACE_DIR_UP_3D,
)

var mesh = BlockMesh3D[P](
    ctx, x_min, x_max, y_min, y_max, z_min, z_max,
    nx=NX, ny=NY, nz=NZ,
    periodic_x=False, periodic_y=True, periodic_z=True,
    bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
)
```

Six face directions, `node_xyz_buf` device buffer.

## Periodic vs non-periodic

`periodic_*=True` configures the face buffers so faces wrap around —
i.e. the right-most face's `face_pos_elem` references element 0, and
vice versa. In that mode the per-side `bc_*` parameters are ignored.

`periodic_*=False` adds an extra row/column of "ghost" faces on the
boundary. The per-axis `bc_*` parameters select what happens at those
ghost faces (see [`core.bc`](core.md)).

## Mesh stores connectivity, not state

The mesh never holds physics state — that's `variable.DGState{1,2,3}D`.
A mesh is reused across solves; states are reset per problem.
