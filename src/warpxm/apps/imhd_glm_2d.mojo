# 2D shim around the consolidated GLM-MHD app. See `apps/imhd_glm.mojo`.

from warpxm.apps.imhd_glm import (
    ImhdGLM,
    MHD_GLM_NC,
    IDX_PSI,
)

comptime ImhdGLM2D = ImhdGLM
