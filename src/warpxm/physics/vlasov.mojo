# Kinetic Vlasov-Maxwell: app, phase-space meshes and states, DG solvers,
# moment and source kernels, and steppers. Use with `warpxm.prelude`; add
# `warpxm.physics.maxwell` for the field solver.
#
# The dimension-generic stack (`PhaseMesh`, `PhaseState`, `DGVlasovSolver`,
# `VlasovMoments`, `apply_vlasov_current_source`, `SSPRK3Phase`,
# `SSPRK3VlasovMaxwell`, all parameterised by `[XDIM, VDIM, P_x, P_v]`)
# covers 1D1V up to 3D3V. The `…1D1V` / `…1D2V` names are the original
# per-dimension implementations; BGK and the 5-moment initialiser exist
# only there.

from warpxm.apps.vlasovmaxwell import VlasovMaxwell, VLASOV_NC
from warpxm.core.bc import BC_VLASOV_OUTFLOW, BC_VLASOV_REFLECT
from warpxm.mesh.phase_mesh import PhaseMesh
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.spatial.kinetic.dg_vlasov import DGVlasovSolver
from warpxm.spatial.kinetic.dg_vlasov_1d1v import DGVlasovSolver1D1V
from warpxm.spatial.kinetic.dg_vlasov_1d2v import DGVlasovSolver1D2V
from warpxm.spatial.kinetic.vlasov_5moment_to_kinetic_1d1v import (
    apply_5moment_to_kinetic_1d1v,
)
from warpxm.spatial.kinetic.vlasov_bgk_source_1d1v import VlasovBGKSource1D1V
from warpxm.spatial.kinetic.vlasov_current_source_1d import (
    apply_vlasov_current_source_1d,
)
from warpxm.spatial.kinetic.vlasov_current_source_1d2v import (
    apply_vlasov_current_source_1d2v,
)
from warpxm.spatial.kinetic.vlasov_current_source import (
    apply_vlasov_current_source,
)
from warpxm.spatial.kinetic.vlasov_moments import (
    VlasovMoments, VLASOV_MOMENT_NC,
    MOM_DENSITY, MOM_FLUX_X, MOM_FLUX_Y, MOM_FLUX_Z, MOM_ENERGY,
)
from warpxm.spatial.kinetic.vlasov_moments_1d1v import VlasovMoments1D1V
from warpxm.spatial.kinetic.vlasov_moments_1d2v import VlasovMoments1D2V
from warpxm.temporal.ssprk3_1d1v import SSPRK3_1D1V, no_op_adjust_1d1v
from warpxm.temporal.ssprk3_1d2v import SSPRK3_1D2V, no_op_adjust_1d2v
from warpxm.temporal.ssprk3_phase import SSPRK3Phase, no_op_adjust_phase
from warpxm.temporal.ssprk3_vlasov_maxwell import SSPRK3VlasovMaxwell
from warpxm.temporal.ssprk3_vlasov_maxwell_1d1v import SSPRK3_VlasovMaxwell_1D1V
from warpxm.temporal.ssprk3_vlasov_maxwell_1d2v import SSPRK3_VlasovMaxwell_1D2V
from warpxm.variable.phase_state import PhaseState
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
