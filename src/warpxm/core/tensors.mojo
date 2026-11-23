# Kernel-argument tensor types shared by every launcher.
#
# All device buffers here are flat and runtime-sized, so one layout type
# serves every buffer length and a kernel signature can be written without
# layout parameters.

from layout import TileTensor, row_major

from warpxm.core.types import WXM_DTYPE, INDEX_DTYPE


comptime FlatLayout = type_of(row_major(Int(0)))
"""Layout of every flat device buffer: rank 1, runtime length."""

comptime RealTensor = TileTensor[WXM_DTYPE, FlatLayout, MutAnyOrigin]
comptime IndexTensor = TileTensor[INDEX_DTYPE, FlatLayout, MutAnyOrigin]
