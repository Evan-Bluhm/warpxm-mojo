# Convergence-rate check shared by the `test_*_convergence_*` tests.

from std.math import log2
from std.testing import assert_true

from warpxm.core.types import Real


def check_convergence(
    label: String,
    err_coarse: Real,
    err_fine: Real,
    expected_rate: Real,
    tol_coarse: Real,
    tol_fine: Real,
    rate_slack: Real = 0.25,
) raises -> Real:
    """Assert the observed order between two meshes (the fine one having
    twice the resolution) and bound both errors absolutely. Returns the
    observed rate.

    Checking only the rate misses a defect that scales the error uniformly;
    set `tol_coarse` / `tol_fine` to a small multiple (about 3x) of the
    measured errors to catch that too.
    """
    var rate = log2(err_coarse / err_fine)
    print(label, " err_coarse =", err_coarse, " err_fine =", err_fine, " rate =", rate)
    assert_true(err_coarse < tol_coarse, label + ": coarse-mesh error above tolerance")
    assert_true(err_fine < tol_fine, label + ": fine-mesh error above tolerance")
    assert_true(
        rate > expected_rate - rate_slack,
        label + ": convergence rate below tolerance",
    )
    return rate
