# 1D shim around the consolidated Euler app. See `apps/euler.mojo` for
# the canonical implementation. `Euler1D` is an alias for `Euler` so the
# 1D DG kernel and tests continue to compile unchanged.

from warpxm.apps.euler import Euler

comptime Euler1D = Euler
