# 3D shim around the consolidated GLM-MHD app. See `apps/imhd_glm.mojo`.
# `MHD_GLM_NC_3D` and `IDX_PSI_3D` are kept as aliases for callers that
# imported the per-dim names.

from warpxm.apps.imhd_glm import (
    ImhdGLM,
    MHD_GLM_NC,
    IDX_PSI,
)

comptime ImhdGLM3D = ImhdGLM
comptime MHD_GLM_NC_3D = MHD_GLM_NC
comptime IDX_PSI_3D = IDX_PSI
