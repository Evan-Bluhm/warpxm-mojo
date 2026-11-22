# Fluid plasma (5-moment Euler species coupled to Maxwell): both apps and
# their solvers, the Lorentz and current coupling sources, and the joint
# steppers. Use with `warpxm.prelude`.
#
# Euler's energy index is `IDX_E`; the Maxwell field indices are
# `IDX_EX .. IDX_BZ`, so the two sets do not collide.

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.io.vtu_presets import dump_euler_frame, dump_maxwell_frame
from warpxm.spatial.dg_euler import DGEulerSolver
from warpxm.spatial.dg_maxwell import DGMaxwellSolver
from warpxm.spatial.sources.lorentz_source import apply_lorentz_source
from warpxm.spatial.sources.maxwell_current_source import apply_current_source
from warpxm.temporal.ssprk3_plasma import SSPRK3_Plasma
from warpxm.temporal.ssprk3_twofluid import SSPRK3_TwoFluid
