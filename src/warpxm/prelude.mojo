# Everything a driver needs that is not specific to one physics.
#
#     from max.gpu.host import DeviceContext
#     from warpxm.prelude import *
#     from warpxm.physics.euler import *
#
# brings in the scalar type and constants, the BC kinds, the meshes, the
# state, the time steppers and the output helpers. The per-physics modules
# in `warpxm/physics/` add the app, its index constants, its solvers and
# its adjusters.

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import (
    BC_COPYOUT, BC_WALL_FREESLIP, BC_PEC_WALL, BC_PMC_WALL,
)
from warpxm.core.math_compat import fabs, fmax, fmin, sign, sqrt, square
from warpxm.core.types import Real, PI, TWO_PI, HALF_PI
from warpxm.io.frame_writer import FrameWriter
from warpxm.io.vtu import write_vtu
from warpxm.mesh.block import BlockMesh
from warpxm.parallel.mpi import (
    is_root, num_ranks, rank_device_context, root_print, global_sum,
    global_min, global_max,
)
from warpxm.temporal.dormand_prince import DormandPrince45
from warpxm.temporal.ssprk3 import SSPRK3, no_op_adjust
from warpxm.variable.state import DGState
