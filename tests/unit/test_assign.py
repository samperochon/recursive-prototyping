"""Unit tests for recursive_prototyping.symbolization.assign."""

import numpy as np
import pytest

from recursive_prototyping.symbolization.assign import (
    assign_to_prototypes,
    background_filter,
)

# Shared hand-built background_filter case, groups interleaved on purpose.
# Group 0 (indices 0, 2, 4, 6): d = [0.1, 0.1, 0.1, 0.5] -> m_0 = 0.2,
#   sigma_0 = sqrt(0.03) ~= 0.1732, threshold ~= 0.3732 -> only 0.5 crosses.
# Group 1 (indices 1, 3, 5, 7, 8): d = [0.2, 0.6, 0.2, 0.2, 1.0] -> m_1 = 0.44,
#   sigma_1 = sqrt(0.512 / 5) = 0.32, threshold = 0.76 -> only 1.0 crosses.
# Group 0 filters d = 0.5 while group 1 keeps d = 0.6 > 0.5: no single global
# cutoff could produce this pattern -- the threshold is per-prototype.
_BG_LABELS = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int64)
_BG_DISTANCES = np.array([0.1, 0.2, 0.1, 0.6, 0.1, 0.2, 0.5, 0.2, 1.0])
_BG_EXPECTED = np.array([0, 1, 0, 1, 0, 1, -1, 1, -1], dtype=np.int64)


def _orthonormal_vocabulary_and_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3 orthonormal prototypes in R^4 (standard basis vectors) plus one
    hand-built small perturbation of each, with the expected labels."""
    vocabulary = np.eye(3, 4)
    X = np.array(
        [
            [1.0, 0.1, 0.0, 0.0],  # prototype 0, tilted toward e1
            [0.0, 1.0, 0.0, -0.2],  # prototype 1, tilted along e3 (outside the vocabulary span)
            [0.15, 0.0, 1.0, 0.0],  # prototype 2, tilted toward e0
        ]
    )
    expected_labels = np.array([0, 1, 2], dtype=np.int64)
    return vocabulary, X, expected_labels


def test_assign_to_prototypes_exact_labels_and_distances():
    """Perturbed copies of orthonormal prototypes get exact labels, with
    distances matching 1 - cos(angle) recomputed independently via numpy."""
    vocabulary, X, expected_labels = _orthonormal_vocabulary_and_points()

    labels, distances = assign_to_prototypes(X, vocabulary)

    np.testing.assert_array_equal(labels, expected_labels)
    expected_distances = np.array(
        [
            1.0 - X[i] @ vocabulary[g] / (np.linalg.norm(X[i]) * np.linalg.norm(vocabulary[g]))
            for i, g in enumerate(expected_labels)
        ]
    )
    np.testing.assert_allclose(distances, expected_distances, atol=1e-6)


def test_assign_to_prototypes_self_normalizes_inputs():
    """Row-scaled (un-normalized) inputs give the same labels and distances
    as pre-normalized ones -- the underlying primitive self-normalizes."""
    vocabulary, X, _ = _orthonormal_vocabulary_and_points()
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    row_scales = np.array([5.0, 0.3, 12.0])

    labels_scaled, distances_scaled = assign_to_prototypes(
        X * row_scales[:, None], vocabulary * 7.0
    )
    labels_normalized, distances_normalized = assign_to_prototypes(X_normalized, vocabulary)

    np.testing.assert_array_equal(labels_scaled, labels_normalized)
    np.testing.assert_allclose(distances_scaled, distances_normalized, atol=1e-6)


def test_assign_to_prototypes_output_dtypes():
    """Labels come back int64 and distances float64."""
    vocabulary, X, _ = _orthonormal_vocabulary_and_points()

    labels, distances = assign_to_prototypes(X, vocabulary)

    assert labels.dtype == np.int64
    assert distances.dtype == np.float64


def test_assign_to_prototypes_rejects_bad_shapes():
    """1-D inputs, an empty vocabulary, and mismatched dims raise ValueError."""
    vocabulary, X, _ = _orthonormal_vocabulary_and_points()

    with pytest.raises(ValueError):
        assign_to_prototypes(X[0], vocabulary)  # 1-D X
    with pytest.raises(ValueError):
        assign_to_prototypes(X, vocabulary[0])  # 1-D vocabulary
    with pytest.raises(ValueError):
        assign_to_prototypes(X, np.empty((0, 4)))  # empty vocabulary
    with pytest.raises(ValueError):
        assign_to_prototypes(X, np.eye(3, 5))  # embedding-dim mismatch


def test_background_filter_per_prototype_thresholds_hand_computed():
    """Exactly one hand-picked outlier per prototype crosses its own
    m_g + sigma_g threshold and becomes -1; everything else is kept (see
    the hand math above _BG_LABELS)."""
    result = background_filter(_BG_LABELS, _BG_DISTANCES, n_sigma=1.0)

    np.testing.assert_array_equal(result, _BG_EXPECTED)


def test_background_filter_boundary_equality_not_filtered():
    """d exactly equal to m_g + n_sigma*sigma_g is kept (strict >).

    Distances [0.25, 0.75] are exact binary fractions: m = 0.5, population
    sigma = 0.25, threshold = 0.75 exactly -- the far window sits exactly
    on the threshold and must survive."""
    labels = np.array([0, 0], dtype=np.int64)
    distances = np.array([0.25, 0.75])

    result = background_filter(labels, distances, n_sigma=1.0)

    np.testing.assert_array_equal(result, labels)


def test_background_filter_n_sigma_zero_thresholds_at_the_mean():
    """With n_sigma=0 the threshold is m_g itself: [0.25, 0.75] has
    m = 0.5, so the far window is now filtered and the near one kept."""
    labels = np.array([0, 0], dtype=np.int64)
    distances = np.array([0.25, 0.75])

    result = background_filter(labels, distances, n_sigma=0.0)

    np.testing.assert_array_equal(result, np.array([0, -1], dtype=np.int64))


def test_background_filter_n_sigma_controls_strictness():
    """On the shared case, n_sigma=0 filters everything strictly above each
    group mean (m_0 = 0.2, m_1 = 0.44), and a large n_sigma filters nothing
    (thresholds 0.2 + 10*0.1732 ~= 1.93 and 0.44 + 10*0.32 = 3.64)."""
    result_tight = background_filter(_BG_LABELS, _BG_DISTANCES, n_sigma=0.0)
    np.testing.assert_array_equal(
        result_tight, np.array([0, 1, 0, -1, 0, 1, -1, 1, -1], dtype=np.int64)
    )

    result_loose = background_filter(_BG_LABELS, _BG_DISTANCES, n_sigma=10.0)
    np.testing.assert_array_equal(result_loose, _BG_LABELS)


def test_background_filter_single_member_prototype_never_filtered():
    """A prototype with one assigned window has sigma_g = 0 and d == m_g,
    so even a huge distance survives -- at n_sigma=1 and even n_sigma=0."""
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    distances = np.array([0.1, 0.1, 0.1, 1.9])

    np.testing.assert_array_equal(background_filter(labels, distances, n_sigma=1.0), labels)
    np.testing.assert_array_equal(background_filter(labels, distances, n_sigma=0.0), labels)


def test_background_filter_does_not_mutate_inputs():
    """The result is a new array and both inputs are byte-identical after
    the call, even though the call did relabel windows."""
    labels = _BG_LABELS.copy()
    distances = _BG_DISTANCES.copy()

    result = background_filter(labels, distances, n_sigma=1.0)

    assert result is not labels
    assert np.any(result != labels)  # the call did filter something
    np.testing.assert_array_equal(labels, _BG_LABELS)
    np.testing.assert_array_equal(distances, _BG_DISTANCES)


def test_background_filter_rejects_already_filtered_labels():
    """Any negative label (already-filtered input) raises ValueError."""
    with pytest.raises(ValueError):
        background_filter(np.array([0, -1, 1]), np.array([0.1, 0.2, 0.3]))


def test_background_filter_empty_arrays():
    """Empty inputs return an empty int64 array without crashing."""
    result = background_filter(np.array([], dtype=np.int64), np.array([]))

    assert result.shape == (0,)
    assert result.dtype == np.int64


def test_background_filter_returns_int64_labels():
    """Output labels are int64 regardless of the input label dtype."""
    result = background_filter(np.array([0, 1], dtype=np.int32), np.array([0.5, 0.5]))

    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, np.array([0, 1], dtype=np.int64))


def test_assign_maps_through_the_nearest_raw_centroid_not_the_prototype_mean():
    # Prototype 0 groups two raw centroids at +-60 degrees around e0; prototype 1 is the single
    # raw centroid at 20 degrees. x sits at 50 degrees: its nearest RAW centroid is the 60-degree
    # one (prototype 0), although prototype 0's MEAN direction (0 degrees) is further from x
    # than prototype 1 (20 degrees). The paper assigns y = Phi(argmin_k d(x, mu_k)).
    def unit(deg):
        rad = np.deg2rad(deg)
        return [np.cos(rad), np.sin(rad)]

    raw_centroids = np.array([unit(60), unit(-60), unit(20)])
    centroid_to_prototype = np.array([0, 0, 1])
    X = np.array([unit(50)])

    labels, distances = assign_to_prototypes(X, raw_centroids, centroid_to_prototype)

    np.testing.assert_array_equal(labels, [0])
    np.testing.assert_allclose(distances, [1 - np.cos(np.deg2rad(10))], atol=1e-6)


def test_assign_rejects_a_mapping_of_the_wrong_length():
    with pytest.raises(ValueError, match="centroid_to_prototype"):
        assign_to_prototypes(np.eye(2), np.eye(2), np.array([0]))
