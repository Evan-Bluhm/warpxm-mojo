# HLLD flux for MHD

The HLLD (**H**arten-**L**ax-**v**an **L**eer-**D**iscontinuities)
Riemann solver for ideal MHD by Miyoshi & Kusano (2005). A five-state
approximation that resolves the **fast magnetosonic**, **Alfvén**, and
**contact** waves — three more wave families than Rusanov / HLL.

Citation: Miyoshi, T. & Kusano, K., "A multi-state HLL approximate
Riemann solver for ideal magnetohydrodynamics", *J. Comput. Phys.*
208 (2005) 315–344.

## Why it matters

Rusanov collapses every Riemann fan into a single dissipative jump. On
problems with rotational discontinuities (e.g. Brio-Wu, Orszag-Tang)
that smears the slow / Alfvén waves heavily. HLLD distinguishes the
five wave regions:

```text
                                          time
                                           ▲
                          ╲     ╱      ╲    │    ╱      ╲     ╱
                         S_L  S_L*    S_M   │  S_R*    S_R
                           ╲   ╲       │    │   │       ╱   ╱
                            ╲   ╲      │    │   │      ╱   ╱
                             ╲   ╲     │    │   │     ╱   ╱
                              ╲   ╲    │    │   │    ╱   ╱
                               ╲   ╲   │    │   │   ╱   ╱
                                ╲   ╲  │    │   │  ╱   ╱
                  q_L            ╲   ╲ │    │   │ ╱   ╱            q_R
                                  ╲   ╲│    │   │╱   ╱
            ── ── ── ── ── ── ── ── ── ●────●────● ── ── ── ── ── ── ── ── ──  x
                                       0
```

* $S_L, S_R$ — outermost fast magnetosonic waves.
* $S_L^*, S_R^*$ — Alfvén waves bracketing the contact.
* $S_M$ — middle / contact wave, has constant total pressure and
  constant normal velocity across it.

In each region the flux comes from a different intermediate "star" or
"double-star" state; HLLD chooses the appropriate one based on where
$s = 0$ falls in the wave fan.

## In code

```mojo
var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)

@parameter
def compute_rhs(mut q_in, mut q_out) raises:
    spatial.compute_rhs(ctx, mesh, q_in, q_out)

@parameter
def adjust(mut q_) raises:
    limiter.apply(ctx, mesh, q_)

stepper.step[compute_rhs, adjust](ctx, q, dt)
```

`USE_HLLD: Bool` is a `comptime` parameter on the DG solver only. At
`False` (the default) the spatial RHS uses
`IMHD.numerical_flux_static` (Rusanov); at `True` it uses
`IMHD.numerical_flux_hlld_static`. The flux choice is captured by the
`compute_rhs` closure, so the stepper doesn't need to know about it —
zero runtime cost.

The 3D version, `DGImhdSolver3D[P, USE_HLLD=True]`, uses a permutation-
based core (`_hlld_core_3d` in `apps/imhd_3d.mojo`) so all three axis-
aligned face directions share one ~150-line implementation.

## Algorithm sketch

For a face with normal $\hat{n}$, the steps are:

1. Rotate $\mathbf{q}_L, \mathbf{q}_R$ into the face-normal frame
   $(n, t_1, t_2)$.
2. Compute fast-wave estimates $S_L, S_R$ (Davis 1988 bounds).
3. Solve a Rankine-Hugoniot condition across the contact to get $S_M$
   and the total pressure $p_T^{*}$ in the star regions.
4. Compute star densities $\rho_{L,R}^{*}$ and Alfvén speeds $S_{L,R}^{*}
   = S_M \mp |B_n|/\sqrt{\rho_{L,R}^{*}}$.
5. Compute star transverse velocities and B-fields (or fall back to a
   simpler form when $B_n \to 0$).
6. Compute the double-star state across the Alfvén waves.
7. Pick the region containing $s = 0$ and return that flux, rotated
   back to $(x, y, z)$.

## Degenerate cases

When $B_n$ is small the denominators in the star-state formulas blow up.
We guard against this with a $|B_n^{2}|$ floor of $10^{-30}$ and fall
back to the HLL-style averaging in that regime — the resulting flux is
identical to HLLD where $B_n$ is significant and degrades gracefully
where it isn't.

## Validation

The 2D and 3D Brio-Wu shock tubes (`tests/test_imhd_briowu_*_hlld.mojo`)
verify HLLD matches the Rusanov far-field state tolerances while giving
sharper near-discontinuity structure. On the 2D Brio-Wu test, the
maximum density gradient `|Δρ/Δx|` peaks at ~0.17 with HLLD vs ~0.10
with Rusanov at the same mesh resolution.
