"""Unit tests for recursive_prototyping.segmentation.changepoint.

Note on gamma in the recovery tests: the per-participant median-heuristic
gamma is very flat on near-noiseless synthetic step
signals -- the kernel cost of the true boundary
and of a boundary shifted by 3 differ only at the ~1e-15 level, so exact
recovery under the auto-gamma path is numerically undefined there. Boundary
recovery is therefore asserted with an explicit, sharper gamma (and with
the linear kernel); the auto-gamma path is exercised by the convention,
penalty-extreme, and explicit-equals-auto tests below.
"""

import numpy as np
import pytest

from recursive_prototyping.config.schema import SegmentationConfig
from recursive_prototyping.segmentation.changepoint import (
    fit_kernel_cpd,
    segment_fixed_m,
    segment_with_penalty,
)
from recursive_prototyping.segmentation.kernel import median_heuristic_gamma

# Sharp explicit RBF bandwidth for boundary-recovery assertions (see module
# docstring); it recovers the true boundaries exactly.
_RBF_TEST_GAMMA = 0.1


def _piecewise_constant_signal(
    T: int = 60, dim: int = 3, seed: int = 47
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-build a piecewise-constant signal in R^dim with 3 well-separated
    level shifts at known positions, small noise, and true change points in
    the repo cpts convention [0, t1, t2, t3, T]."""
    rng = np.random.default_rng(seed)
    true_cpts = np.array([0, 15, 30, 45, T], dtype=np.int64)
    levels = [0.0, 5.0, -5.0, 10.0]

    X = np.zeros((T, dim))
    for (start, end), level in zip(zip(true_cpts[:-1], true_cpts[1:]), levels):
        X[start:end] = level
    X += rng.normal(scale=0.1, size=(T, dim))

    return X, true_cpts


def test_segment_fixed_m_rbf_recovers_true_boundaries():
    X, true_cpts = _piecewise_constant_signal()
    config = SegmentationConfig()

    cpts = segment_fixed_m(X, n_cpts=3, config=config, gamma=_RBF_TEST_GAMMA)
    assert len(cpts) == len(true_cpts)
    np.testing.assert_allclose(cpts, true_cpts, atol=1)


def test_segment_fixed_m_linear_kernel_recovers_true_boundaries_exactly():
    X, true_cpts = _piecewise_constant_signal()
    config = SegmentationConfig(kernel="linear")

    cpts = segment_fixed_m(X, n_cpts=3, config=config)
    np.testing.assert_array_equal(cpts, true_cpts)


def test_outputs_follow_repo_cpts_convention():
    X, _ = _piecewise_constant_signal()
    config = SegmentationConfig()

    fixed = segment_fixed_m(X, n_cpts=3, config=config)
    penalized = segment_with_penalty(X, penalty=0.1, config=config, gamma=1.0)

    for cpts in (fixed, penalized):
        assert cpts.dtype == np.int64
        assert cpts[0] == 0
        assert cpts[-1] == X.shape[0]
        assert np.all(np.diff(cpts) > 0)


def test_every_segment_at_least_min_segment_size():
    X, _ = _piecewise_constant_signal()

    config = SegmentationConfig(min_segment_size=5)
    cpts = segment_fixed_m(X, n_cpts=3, config=config)
    assert np.all(np.diff(cpts) >= config.min_segment_size)

    # Tiny penalty + sharp gamma drives the count high enough that the
    # min-size constraint actually binds.
    config_default = SegmentationConfig()
    cpts = segment_with_penalty(X, penalty=1e-4, config=config_default, gamma=1.0)
    assert len(cpts) - 2 >= 20
    assert np.all(np.diff(cpts) >= config_default.min_segment_size)


def test_segment_with_penalty_huge_penalty_yields_no_change_points():
    X, _ = _piecewise_constant_signal()
    config = SegmentationConfig()

    np.testing.assert_array_equal(
        segment_with_penalty(X, penalty=1e9, config=config), [0, X.shape[0]]
    )
    np.testing.assert_array_equal(
        segment_with_penalty(X, penalty=1e9, config=config, gamma=1.0),
        [0, X.shape[0]],
    )


def test_segment_with_penalty_larger_penalty_never_more_change_points():
    X, _ = _piecewise_constant_signal()
    config = SegmentationConfig()

    penalties = [1e-4, 1e-2, 0.1, 1.0, 1e9]
    n_cpts = [
        len(segment_with_penalty(X, penalty=pen, config=config, gamma=1.0)) - 2
        for pen in penalties
    ]

    assert all(a >= b for a, b in zip(n_cpts, n_cpts[1:]))
    assert n_cpts[0] >= 20  # tiny penalty yields many change points
    assert n_cpts[-1] == 0  # huge penalty yields none


def test_unsupported_kernel_raises_value_error():
    X, _ = _piecewise_constant_signal()

    for bad_kernel in ("cosine", "unknown"):
        with pytest.raises(ValueError, match="Unsupported kernel"):
            fit_kernel_cpd(X, SegmentationConfig(kernel=bad_kernel))


def test_explicit_median_heuristic_gamma_matches_auto():
    X, _ = _piecewise_constant_signal()
    config = SegmentationConfig()  # gamma_temperature=1.0
    gamma = median_heuristic_gamma(X, 1.0)

    np.testing.assert_array_equal(
        segment_fixed_m(X, n_cpts=3, config=config, gamma=gamma),
        segment_fixed_m(X, n_cpts=3, config=config, gamma=None),
    )


def test_same_call_twice_is_deterministic():
    X, _ = _piecewise_constant_signal()
    config = SegmentationConfig()

    np.testing.assert_array_equal(
        segment_fixed_m(X, n_cpts=3, config=config),
        segment_fixed_m(X, n_cpts=3, config=config),
    )
    np.testing.assert_array_equal(
        segment_with_penalty(X, penalty=1.0, config=config),
        segment_with_penalty(X, penalty=1.0, config=config),
    )


def test_invalid_inputs_raise_value_error():
    X, _ = _piecewise_constant_signal()
    config = SegmentationConfig()

    for bad_n_cpts in (0, -1):
        with pytest.raises(ValueError, match="n_cpts"):
            segment_fixed_m(X, n_cpts=bad_n_cpts, config=config)

    for bad_penalty in (0.0, -1.0):
        with pytest.raises(ValueError, match="penalty"):
            segment_with_penalty(X, penalty=bad_penalty, config=config)

    with pytest.raises(ValueError, match="2-D"):
        fit_kernel_cpd(X[:, 0], config)


def test_config_jump_is_ignored_by_ruptures_kernel_cpd():
    # ruptures v1.1.10's KernelCPD does not support candidate subsampling:
    # it stores jump=1 no matter what is passed (see fit_kernel_cpd's
    # docstring). Pin that here so a future ruptures upgrade that changes
    # the behavior gets noticed.
    X, _ = _piecewise_constant_signal()
    model = fit_kernel_cpd(X, SegmentationConfig(jump=5))
    assert model.jump == 1
