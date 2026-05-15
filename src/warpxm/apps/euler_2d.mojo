# 2D shim around the consolidated Euler app. The single source of truth
# lives in `apps/euler.mojo`. `Euler2D` is kept as an alias so existing
# callers (44 test / example / spatial files) continue to work.

from warpxm.apps.euler import (
    Euler,
    EULER_NC,
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    _rho_floored, _pressure_floored,
)

comptime Euler2D = Euler
