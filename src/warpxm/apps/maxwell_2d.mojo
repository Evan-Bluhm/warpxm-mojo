# 2D shim around the consolidated Maxwell app. See `apps/maxwell.mojo`.

from warpxm.apps.maxwell import (
    Maxwell,
    MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)

comptime Maxwell2D = Maxwell
