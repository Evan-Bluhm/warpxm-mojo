# Ideal MHD with Dedner GLM divergence cleaning: app, index constants, DG
# solvers, ψ damping source, limiter, positivity preserver and VTU preset.
# Use with `warpxm.prelude`.

from warpxm.adjusters.moe_rossmanith import MoeRossmanith
from warpxm.adjusters.positivity_mhd import PositivityMHD
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC, IDX_PSI
from warpxm.io.vtu_presets import dump_imhd_glm_frame
from warpxm.spatial.dg_imhd_glm import DGImhdGLMSolver
from warpxm.spatial.sources.imhd_glm_source import apply_glm_damping
