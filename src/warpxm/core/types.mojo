# Core scalar / dtype aliases for warpxm-mojo.
#
# Mirrors the role of src/lib/wxm_math_compat.h and the `real` typedef in
# the upstream codebase. Everything is Float64: kernels are written against
# `Real` / `WXM_DTYPE` throughout, but single precision has never been run
# or tested, so treat the alias as a name, not as a supported switch.

# ---------------------------------------------------------------------------
# Working dtype + scalar alias.
# ---------------------------------------------------------------------------

comptime WXM_DTYPE: DType = DType.float64
comptime Real = Scalar[WXM_DTYPE]

# Index dtype used by mesh / connectivity buffers. Matches the upstream
# convention of using 32-bit ints for element / face indexing.
comptime INDEX_DTYPE: DType = DType.int32

# ---------------------------------------------------------------------------
# Numeric constants. Float literals are polymorphic and adapt to Real.
# ---------------------------------------------------------------------------

comptime PI: Real = 3.14159265358979323846
comptime TWO_PI: Real = 2.0 * 3.14159265358979323846
comptime HALF_PI: Real = 0.5 * 3.14159265358979323846

# Small epsilon used by upstream for the absc/dt cutoff in numerical fluxes
# (see e.g. apps/simple/advection.h).
comptime WXM_TINY: Real = 1e-14

# Sentinel used in place of upstream's `wxm::math::infinity()`. Used by
# physics-app `*_flux` returns to mean "no CFL constraint contributed by this
# kernel call." A finite-but-huge value works the same as IEEE inf for the
# `min(dt, sugg_dt)` aggregator while avoiding fragile target-dependent inf
# spellings.
comptime WXM_INF: Real = 1.0e308
