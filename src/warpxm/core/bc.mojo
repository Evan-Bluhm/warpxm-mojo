# Boundary-condition kinds.
#
# `BCKind` enumerates the supported BC types that can sit on a mesh boundary.
# Upstream WARPXM exposes these as runtime-named adjusters (e.g. "copyout",
# "freeslip_wall", "dirichlet"); here they're plain compile-time integer
# constants that flow into the spatial-solver kernel's BC switch.
#
# Today's list is intentionally minimal. The full upstream catalogue includes
# `noslip_wall`, `axis`, `pulse_inlet`, `floquet_bloch`, and several
# physics-specific BCs — these can be added as new constants + new branches in
# the kernel's BC dispatch as the port grows.

# `INTERIOR` is the sentinel for non-boundary faces; the BC table is written
# to this value by mesh construction for any face that isn't a domain
# boundary. The DG kernel ignores BC info entirely for these faces.
comptime BC_INTERIOR: Int = 0

# Zero-gradient / Neumann BC. Ghost state equals the adjacent interior state.
# Acts as a soft outflow for advection-dominated flows; produces no spurious
# wave reflections so long as everything that hits the boundary is leaving.
comptime BC_COPYOUT: Int = 1

# Free-slip (reflecting) wall. Normal velocity component is reversed in the
# ghost state, tangential components are mirrored, scalar variables are
# copied. Used for the "Wall" BC on impermeable boundaries in 5-moment Euler.
comptime BC_WALL_FREESLIP: Int = 2

# Perfect electric conductor (PEC) wall — Maxwell-specific reflecting BC.
# Tangential components of E (and the normal component of B) are zero on
# the wall; the ghost state is built by flipping those components from the
# interior so the face-average matches.
comptime BC_PEC_WALL: Int = 4

# Perfect magnetic conductor (PMC) wall — the dual of PEC. Tangential B and
# normal E are zero on the wall; ghost state flips those components.
comptime BC_PMC_WALL: Int = 5

# Vlasov phase-space outflow — mirrors upstream `outflow_bc_1d1v.{h,cc}`.
# Used on velocity-axis boundaries of a kinetic phase mesh. The physical
# distribution function vanishes outside [v_min, v_max] (the velocity
# domain is a truncation of the real line where f is exponentially
# small), so any flux into the domain across these boundaries should be
# zero. The kernel implements this by setting the ghost state to zero,
# which combined with the upwind numerical flux yields no spurious
# inflow when characteristics point into the domain. Distinct from
# `BC_COPYOUT`, which mirrors the interior and would inject `f_interior`
# back across the truncation boundary.
comptime BC_VLASOV_OUTFLOW: Int = 6
