# `warpxm.prelude` and `warpxm.physics`

Re-export modules. A driver imports the prelude and one physics module
instead of a dozen individual files:

```mojo
from std.collections import Array
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime State = DGState[2, P, EULER_NC]


def main() raises:
    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=64, ny=64, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        return [Real(1.0), Real(0.5), Real(0.0), Real(0.0), Real(2.625)]
    mesh.fill(q, initial)

    var spatial = DGEulerSolver[2, P](
        Euler(gas_gamma=Real(1.4), min_density=Real(1e-12),
              min_pressure=Real(1e-12), cfl_max=Real(0.5)),
    )
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    var t = stepper.run_until_cfl(
        ctx, q, Real(0.0), Real(1.0),
        compute_rhs, no_op_adjust[2, P, EULER_NC], suggest_dt,
    )
    print("t =", t, " mass =", mesh.integrate(q, IDX_RHO))
```

Everything is also importable from its defining module; the pages for
each package give those paths.

## `warpxm.prelude`

Everything not tied to one physics.

| From | Names |
| ---- | ----- |
| `core.types` | `Real`, `PI`, `TWO_PI`, `HALF_PI` |
| `core.math_compat` | `fabs`, `fmax`, `fmin`, `sign`, `sqrt`, `square` |
| `core.bc` | `BC_COPYOUT`, `BC_WALL_FREESLIP`, `BC_PEC_WALL`, `BC_PMC_WALL` |
| `basis` | `NodalBasis1D` |
| `mesh` | `BlockMesh` |
| `variable` | `DGState` |
| `temporal` | `SSPRK3`, `no_op_adjust`, `DormandPrince45` |
| `io` | `FrameWriter`, `write_vtu` |

Not included: `DeviceContext` (from `max.gpu.host`), `Array` (from
`std.collections`), and `std.math` functions such as `sin` and `exp`.

## `warpxm.physics.*`

| Module | Exports |
| ------ | ------- |
| `physics.advection` | `Advection`, `DGAdvectionSolver` |
| `physics.euler` | `Euler`, `EULER_NC`, `IDX_RHO`, `IDX_MX`, `IDX_MY`, `IDX_MZ`, `IDX_E`, `DGEulerSolver`, `MoeRossmanith`, `PositivityEuler`, `apply_gravity_source`, `dump_euler_frame` |
| `physics.maxwell` | `Maxwell`, `MAXWELL_NC`, `IDX_EX` … `IDX_BZ`, `DGMaxwellSolver`, `dump_maxwell_frame` |
| `physics.imhd` | `IMHD`, `MHD_NC`, `IDX_RHO` … `IDX_E`, `IDX_BX`, `IDX_BY`, `IDX_BZ`, `DGImhdSolver`, `MoeRossmanith`, `PositivityMHD`, `dump_imhd_frame` |
| `physics.imhd_glm` | `ImhdGLM`, `MHD_GLM_NC`, the MHD indices plus `IDX_PSI`, `DGImhdGLMSolver`, `apply_glm_damping`, `MoeRossmanith`, `PositivityMHD`, `dump_imhd_glm_frame` |
| `physics.plasma` | everything in `physics.euler`'s and `physics.maxwell`'s app, index and solver lists, plus `apply_lorentz_source`, `apply_current_source`, `SSPRK3_Plasma`, `SSPRK3_TwoFluid`, `dump_euler_frame`, `dump_maxwell_frame` (no limiter or positivity preserver) |
| `physics.vlasov` | `VlasovMaxwell`, `VLASOV_NC`, `BC_VLASOV_OUTFLOW`, `BC_VLASOV_REFLECT`, `PhaseMesh`, `PhaseState`, `DGVlasovSolver`, `VlasovMoments`, `VLASOV_MOMENT_NC`, `MOM_DENSITY`, `MOM_FLUX_X`, `MOM_FLUX_Y`, `MOM_FLUX_Z`, `MOM_ENERGY`, `apply_vlasov_current_source`, `SSPRK3Phase`, `no_op_adjust_phase`, `SSPRK3VlasovMaxwell`, `VlasovBGKSource`, `apply_5moment_to_kinetic` |

## Combining modules

* `physics.euler`, `physics.imhd` and `physics.imhd_glm` all export
  `IDX_RHO` … `IDX_E` (same names, same values) and the limiters.
  Star-import only one of them in a file; import what you need from the
  others by name.
* `physics.plasma` already contains the Euler and Maxwell names, so a
  fluid-plasma driver needs only it. Euler's energy index is `IDX_E`;
  the Maxwell field indices are `IDX_EX` … `IDX_BZ`, so they do not
  collide.
* A kinetic driver uses `physics.vlasov` together with
  `physics.maxwell`.
* The less common names (`BC_INTERIOR`, `WXM_DTYPE`, `INDEX_DTYPE`,
  `WXM_TINY`, `WXM_INF`, the mesh face-direction constants, the launcher
  types) are not re-exported: import them from their modules.
