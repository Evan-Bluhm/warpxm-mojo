# 2D shim around the consolidated IMHD app. See `apps/imhd.mojo` for the
# canonical implementation. The 2D in-plane-rotation HLLD has been
# replaced by axis-aligned permutation dispatch; this is bit-equivalent
# at axis-aligned 2D faces (which is all the DG launcher ever produces).

from warpxm.apps.imhd import (
    IMHD,
    MHD_NC,
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
    _rho_floored, _bsq, _gas_pressure_floored, _fast_speed,
)

comptime IMHD2D = IMHD
