"""Unit tests for recursive_prototyping.segmentation.penalty (pure calibration math).

collect_risk_curve and calibrate_and_segment depend on the ruptures-backed
sibling modules and are covered by the integration
tests, not here.
"""

import numpy as np

from recursive_prototyping.config.schema import SegmentationConfig
from recursive_prototyping.segmentation.penalty import (
    default_m_max,
    estimate_pooled_slope,
    m_grid,
    penalty_from_slope,
    pool_risk_curves,
)

_SLOPE = -0.004
_INTERCEPTS = {"p1": 3.0, "p2": 5.5, "p3": 1.25}
_N_WINDOWS = {"p1": 1000, "p2": 1500, "p3": 800}
_M_MAXES = {"p1": 100, "p2": 150, "p3": 80}


def _linear_risk_curves(noise_scale: float = 0.0, seed: int = 47) -> dict:
    """3 fake participants whose normalized risk is exactly a_p + s*m over their
    own m-grids (plus optional gaussian noise), multiplied back by n_p into the
    raw-risk units pool_risk_curves expects."""
    rng = np.random.default_rng(seed)
    config = SegmentationConfig()
    curves = {}
    for participant_id, intercept in _INTERCEPTS.items():
        m_values = m_grid(_M_MAXES[participant_id], config)
        normalized = intercept + _SLOPE * m_values
        if noise_scale > 0.0:
            normalized = normalized + rng.normal(scale=noise_scale, size=m_values.shape)
        n_windows = _N_WINDOWS[participant_id]
        curves[participant_id] = (m_values, normalized * n_windows, n_windows)
    return curves


def test_default_m_max_thirty_minute_video_is_900():
    # 5625 windows x 0.32 s = 1800 s (~30 min) -> round(1800 / 2.0) = 900.
    # The hop is the VideoMAE-v2 stride (8 frames at 25 fps), NOT the 0.64 s
    # window span.
    config = SegmentationConfig()
    assert default_m_max(5625, config, hop_seconds=0.32) == 900


def test_default_m_max_uses_the_stride_not_the_window_span():
    # The default hop is the 0.32 s stride; the 0.64 s window span would double M_max.
    config = SegmentationConfig()
    assert default_m_max(5625, config) == 900
    assert default_m_max(5625, config, hop_seconds=0.64) == 1800


def test_default_m_max_config_override_respected():
    config = SegmentationConfig(m_max=50)
    assert default_m_max(2812, config) == 50


def test_default_m_max_feasibility_clip_for_tiny_input():
    # Duration-derived candidate round(10 * 10.0 / 2.0) = 50 far exceeds the
    # ruptures-feasible maximum floor(10 / 2) - 1 = 4.
    config = SegmentationConfig()
    assert default_m_max(10, config, hop_seconds=10.0) == 4


def test_default_m_max_override_also_clipped_to_feasibility():
    config = SegmentationConfig(m_max=50)
    assert default_m_max(10, config) == 4


def test_default_m_max_floored_at_two():
    # 6 windows x 0.1 s = 0.6 s -> round(0.3) = 0, raised to the floor of 2
    # (which equals the feasible maximum floor(6 / 2) - 1 = 2).
    config = SegmentationConfig()
    assert default_m_max(6, config, hop_seconds=0.1) == 2


def test_m_grid_spans_low_frac_to_m_max():
    config = SegmentationConfig()
    grid = m_grid(900, config)
    assert grid.dtype == np.int64
    assert np.all(np.diff(grid) > 0)  # ascending and unique
    assert grid[0] == 540  # 0.6 * 900
    assert grid[-1] == 900
    assert len(grid) <= config.m_grid_size


def test_m_grid_small_m_max_collapses_duplicates():
    config = SegmentationConfig()
    np.testing.assert_array_equal(m_grid(3, config), [2, 3])


def test_m_grid_m_max_one_does_not_crash():
    config = SegmentationConfig()
    grid = m_grid(1, config)
    np.testing.assert_array_equal(grid, [1])
    assert np.all(grid >= 1)


def test_pool_risk_curves_columns_and_normalization():
    curves = {
        "a": (np.array([2, 3, 4]), np.array([10.0, 8.0, 7.0]), 5),
        "b": (np.array([3, 4]), np.array([20.0, 18.0]), 10),
    }
    pooled = pool_risk_curves(curves)

    assert list(pooled.columns) == [
        "participant_id",
        "m",
        "risk",
        "n_windows",
        "normalized_risk",
    ]
    assert len(pooled) == 5
    assert list(pooled["participant_id"]) == ["a", "a", "a", "b", "b"]
    np.testing.assert_array_equal(pooled["m"], [2, 3, 4, 3, 4])
    np.testing.assert_array_equal(pooled["n_windows"], [5, 5, 5, 10, 10])
    # normalized_risk = risk / n_windows, checked by hand.
    np.testing.assert_allclose(pooled["normalized_risk"], [2.0, 1.6, 1.4, 2.0, 1.8])


def test_estimate_pooled_slope_noiseless_exact_recovery():
    pooled = pool_risk_curves(_linear_risk_curves())
    fit = estimate_pooled_slope(pooled)

    np.testing.assert_allclose(fit.slope, _SLOPE, rtol=0, atol=1e-10)
    for participant_id, intercept in _INTERCEPTS.items():
        np.testing.assert_allclose(
            fit.intercepts[participant_id], intercept, rtol=0, atol=1e-8
        )
    np.testing.assert_allclose(fit.r_squared, 1.0, atol=1e-8)
    assert fit.n_obs == len(pooled)


def test_estimate_pooled_slope_noisy_recovery():
    pooled = pool_risk_curves(_linear_risk_curves(noise_scale=0.005, seed=47))
    fit = estimate_pooled_slope(pooled)
    np.testing.assert_allclose(fit.slope, _SLOPE, rtol=0, atol=5e-4)


def test_penalty_from_slope_reference_value():
    assert penalty_from_slope(-0.5, 100) == 100.0


def test_penalty_from_slope_positive_for_negative_slope():
    assert penalty_from_slope(-0.004, 2812) > 0.0


def test_penalty_from_slope_applies_factor_of_two_exactly_once():
    # Locks lambda = -2 * n * s with the -2 applied exactly once to the raw
    # slope (see docs/METHOD.md, section 4, "Penalty factor").
    for slope, n_windows in [(-0.004, 2812), (-1.7, 3), (0.25, 10)]:
        assert penalty_from_slope(slope, n_windows) == -2.0 * n_windows * slope
