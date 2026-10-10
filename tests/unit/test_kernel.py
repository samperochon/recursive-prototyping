"""Unit tests for recursive_prototyping.segmentation.kernel."""

import numpy as np
import pytest

from recursive_prototyping.segmentation.kernel import median_heuristic_gamma


def test_median_heuristic_gamma_matches_hand_computed_value():
    # 3 points in R^2 with hand-computed pairwise squared euclidean distances:
    #   d^2(p0, p1) = 1, d^2(p0, p2) = 4, d^2(p1, p2) = 1 + 4 = 5
    # => sqrt distances (1, 2, sqrt(5)), median sigma_rbf = 2,
    # => gamma = 1 / (2 * temperature * sigma_rbf^2) = 1 / (2 * 1 * 4) = 1/8.
    # There is deliberately NO sequence-length term: the temperature is a scalar,
    # not the number of rows.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    gamma = median_heuristic_gamma(X)
    np.testing.assert_allclose(gamma, 1.0 / 8.0)


def test_median_heuristic_gamma_is_independent_of_sequence_length():
    # Duplicating the rows leaves the median pairwise distance unchanged, so
    # gamma must not move. A formula with the number of rows in the
    # denominator would halve it.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    X_doubled = np.vstack([X, X + 1e-9])

    np.testing.assert_allclose(
        median_heuristic_gamma(X_doubled), median_heuristic_gamma(X), rtol=1e-6
    )


def test_median_heuristic_gamma_temperature_divides_resulting_gamma():
    # The temperature is in the DENOMINATOR: raising it lowers gamma
    # (wider, smoother kernel). gamma = 1/(2*temperature*sigma^2).
    rng = np.random.default_rng(47)
    X = rng.normal(size=(10, 4))

    np.testing.assert_allclose(
        median_heuristic_gamma(X, temperature=2.0),
        median_heuristic_gamma(X, temperature=1.0) / 2.0,
    )


def test_median_heuristic_gamma_rejects_nonpositive_temperature():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        median_heuristic_gamma(X, temperature=0.0)


def test_median_heuristic_gamma_differs_across_participants():
    # Per-participant property: different sequences give different
    # gamma. Scaling X by 2 scales sigma_rbf by 2, hence gamma by 1/4.
    X_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    X_b = 2.0 * X_a

    gamma_a = median_heuristic_gamma(X_a)
    gamma_b = median_heuristic_gamma(X_b)
    assert gamma_a != gamma_b
    np.testing.assert_allclose(gamma_b, gamma_a / 4.0)


def test_median_heuristic_gamma_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        median_heuristic_gamma(np.array([1.0, 2.0, 3.0]))


def test_median_heuristic_gamma_rejects_single_row():
    with pytest.raises(ValueError, match="at least 2 rows"):
        median_heuristic_gamma(np.array([[1.0, 2.0, 3.0]]))


def test_median_heuristic_gamma_rejects_identical_rows():
    with pytest.raises(ValueError, match="identical"):
        median_heuristic_gamma(np.ones((4, 3)))


def test_median_heuristic_gamma_returns_finite_plain_float():
    rng = np.random.default_rng(47)
    X = rng.normal(size=(10, 4))

    gamma = median_heuristic_gamma(X)
    assert type(gamma) is float
    assert np.isfinite(gamma)
    assert gamma > 0.0
