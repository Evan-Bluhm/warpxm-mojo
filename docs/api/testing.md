# `warpxm.testing`

Code shared by the regression tests. It lives in the library so each test
does not carry its own copy.

## `ExactRiemann`

Exact solution of the 1D Euler Riemann problem for an ideal gas (Toro,
*Riemann Solvers and Numerical Methods for Fluid Dynamics*, §4.2–4.5).
Reference for the Sod, Lax and Einfeldt 1-2-3 tests.

```mojo
from warpxm.testing.riemann import ExactRiemann

#                        gamma  rho_l  u_l   p_l   rho_r  u_r   p_r
var exact = ExactRiemann(GAMMA, 1.0,   0.0,  1.0,  0.125, 0.0,  0.1)

var w = exact.sample((x - x0) / t)    # Array[Real, 3]: (rho, u, p)
print(exact.p_star, exact.u_star)
```

| Member | Meaning |
| ------ | ------- |
| `ExactRiemann(gamma, rho_l, u_l, p_l, rho_r, u_r, p_r)` | solves for the star state by bisection (100 iterations) |
| `p_star`, `u_star` | pressure and velocity between the two nonlinear waves |
| `sample(xi)` | primitive state at the similarity coordinate $\xi = (x - x_0)/t$ |

It handles shocks and rarefactions on either side. It does not treat the
vacuum-generating case (it assumes a positive star pressure): the
Einfeldt 1-2-3 problem used in the tests stays above vacuum.

## `check_convergence`

```mojo
from warpxm.testing.convergence import check_convergence

var rate = check_convergence(
    "P = 2", err_coarse, err_fine,
    expected_rate=Real(3.0), tol_coarse=Real(2.5e-3), tol_fine=Real(3.0e-4),
)
```

Given the errors on two meshes, the fine one with twice the resolution,
it prints both and the observed rate `log2(err_coarse / err_fine)`, then
asserts

* `err_coarse < tol_coarse` and `err_fine < tol_fine`,
* `rate > expected_rate - rate_slack` (`rate_slack` defaults to 0.25),

and returns the rate. The absolute bounds matter: a defect that scales
the error uniformly leaves the rate intact. Set them to about three
times the measured errors.

## Related

The mesh helpers (`fill`, `integrate`, `l2_error`, `visit_nodes`) that
tests use for initial conditions and diagnostics are part of the mesh
API: see [`mesh`](mesh.md#host-side-helpers). The suite itself is
described in [Tests](../contributing/tests.md).
