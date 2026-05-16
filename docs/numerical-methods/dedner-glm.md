# Dedner GLM divergence cleaning

A scalar transport equation for $\psi$ added to the MHD system,
producing damped wave transport of $\nabla\cdot\mathbf{B}$ errors out of
the domain. Used by every example whose name contains `imhd_glm` or
`orszag_tang`.

Citation: Dedner, A., Kemm, F., Kröner, D., Munz, C.-D., Schnitzer, T.
& Wesenberg, M., "Hyperbolic divergence cleaning for the MHD
equations", *J. Comput. Phys.* 175 (2002) 645–673.

## What's added

The "MixedGLM" form augments MHD with $\psi$:

$$
\partial_t\mathbf{B} = -\nabla\times\mathbf{E} - \nabla\psi,
$$

$$
\partial_t\psi + c_h^{2}\,\nabla\cdot\mathbf{B} = -\tfrac{c_h^{2}}{c_p^{2}}\psi.
$$

$\psi$ is **not** a physical quantity — it's a Lagrange multiplier
that carries the divergence error away at speed $c_h$ and damps it on
a timescale $c_p^{2}/c_h^{2}$.

## Parameters

```mojo
var app = ImhdGLM(
    gas_gamma=Real(5.0/3.0),
    min_density=Real(1e-12),
    min_pressure=Real(1e-12),
    cfl_max=Real(0.5),
    c_h=Real(4.0),      # cleaning-wave speed
    c_p_sq=Real(0.5),   # damping timescale²
)
```

Choosing parameters:

* $c_h$ should exceed the fastest physical wave speed (fast
  magnetosonic). Otherwise the cleaning waves slow down information
  propagation.
* $c_p^{2}$ controls how aggressively $\psi$ damps. Smaller =
  stronger damping. We use $c_p^{2} = 0.5$ as a starting point.

## State layout

GLM-MHD adds one scalar — `MHD_GLM_NC = 9`, with $\psi$ at index 8:

```mojo
from warpxm.apps.imhd_glm import MHD_GLM_NC, IDX_PSI
# IDX_PSI = 8
```

All other indices are the same as plain ideal MHD.

## Why it works

The cleaning system has the property that, if the initial $\psi = 0$
and $\nabla\cdot\mathbf{B} = 0$, both remain zero in the absence of
discretisation error. When discretisation produces $\nabla\cdot\mathbf{B}
\neq 0$ in some region, $\psi$ acquires a value that propagates the
error away at speed $c_h$ and dissipates it.

The implementation is a single additional source kernel (`apply_glm_damping_*`)
that adds the damping term to the $\psi$ RHS at every RK stage, plus an
extra "ψ-coupled-B" flux contribution baked into the GLM-MHD spatial
solver.

## Validation

The 2D and 3D `test_imhd_glm_decay_*.mojo` tests inject a Gaussian
$\psi$ bump on an otherwise-divergence-free background. Over $t = 4$
time units the integrated $\psi^{2}$ decays by a factor of ~45000.

The Orszag-Tang vortex example (`examples/imhd_orszag_tang_2d_vtu.mojo`)
runs with GLM cleaning and reports the integrated $\psi^{2}$ each
frame; it stays bounded (the discretisation error feeds in, the damping
removes it).

## When you don't need it

If the IC has $\nabla\cdot\mathbf{B} = 0$ analytically *and* the
discretisation can't produce divergence error (uniform B-field, 1D-
extruded problems), you can use plain ideal MHD. The 3D blast-wave
test confirms this — with $\mathbf{B}_0 = (B_0, 0, 0)$ uniform, $\int
B_x\,dV$ stays preserved to 4 × 10⁻¹⁵ without cleaning.

For shocks in non-trivial topologies you'll want GLM. The Orszag-Tang
vortex without GLM diverges within ~0.1 time units; with GLM it runs
to $t = 0.5$ with mass drift 6 × 10⁻¹⁴.
