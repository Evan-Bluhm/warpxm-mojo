# Cell-mean quadrature weight shared by the adjuster kernels.

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE


@always_inline
def gll_cell_weight[P: Int, NDIM: Int](n: Int) -> Scalar[WXM_DTYPE]:
    """Weight of local node `n` in the cell mean: the tensor-product GLL
    weight on the reference element `[-1, 1]^NDIM`, normalised by
    `1 / 2^NDIM` so the weights of one element sum to 1."""
    comptime P1 = P + 1
    var weights = NodalBasis1D[P].weights()
    comptime if NDIM == 1:
        return weights[n] * Real(0.5)
    comptime if NDIM == 2:
        var jn = n // P1
        var in_ = n % P1
        return weights[in_] * weights[jn] * Real(0.25)
    # NDIM == 3
    var kn = n // (P1 * P1)
    var rem = n % (P1 * P1)
    var jn = rem // P1
    var in_ = rem % P1
    return weights[in_] * weights[jn] * weights[kn] * Real(0.125)
