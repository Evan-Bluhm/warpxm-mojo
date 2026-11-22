# Maxwell: app, index constants, DG solvers and VTU preset. Use with
# `warpxm.prelude`.

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.io.vtu_presets import dump_maxwell_frame
from warpxm.spatial.dg_maxwell import DGMaxwellSolver
