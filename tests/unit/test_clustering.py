"""Unit tests for recursive_prototyping.prototyping.clustering."""

from collections import Counter

import numpy as np

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.clustering import (
    assign_to_centroids,
    fit_cosine_kmeans,
    l2_normalize,
)


def _three_well_separated_clusters(
    n_per_cluster: int = 30, dim: int = 8, seed: int = 47
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-build 3 well-separated unit-vector clusters in R^dim.

    Base directions are the first 3 standard basis vectors of R^dim
    (mutually orthogonal), each perturbed by small Gaussian noise and
    renormalized to unit length -- independent of, but structurally
    similar to, the synthetic generator's construction.
    """
    rng = np.random.default_rng(seed)
    base_directions = np.eye(3, dim)

    points, true_labels = [], []
    for cluster_idx, direction in enumerate(base_directions):
        noise = rng.normal(scale=0.05, size=(n_per_cluster, dim))
        points.append(l2_normalize(direction + noise))
        true_labels.extend([cluster_idx] * n_per_cluster)

    X = np.vstack(points)
    return X, np.array(true_labels, dtype=np.int64)


def _cluster_purity(labels: np.ndarray, true_labels: np.ndarray, n_clusters: int) -> float:
    """Majority-overlap purity: for each recovered cluster, count the size of its
    majority true-label group, sum over clusters, divide by N. Robust to the
    arbitrary ordering of k-means cluster indices."""
    correct = 0
    for cluster_idx in range(n_clusters):
        mask = labels == cluster_idx
        if not np.any(mask):
            continue
        correct += Counter(true_labels[mask]).most_common(1)[0][1]
    return correct / len(labels)


def _fast_config(**overrides) -> PrototypingConfig:
    defaults = dict(faiss_niter=50, faiss_nredo=2, faiss_seed=47)
    defaults.update(overrides)
    return PrototypingConfig(**defaults)


def test_fit_cosine_kmeans_recovers_well_separated_clusters():
    X, true_labels = _three_well_separated_clusters()
    config = _fast_config()

    centroids, kmeans = fit_cosine_kmeans(X, n_clusters=3, config=config)
    assert centroids.shape == (3, X.shape[1])
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)

    labels, _ = assign_to_centroids(X, centroids)
    purity = _cluster_purity(labels, true_labels, n_clusters=3)
    assert purity >= 0.95


def test_fit_cosine_kmeans_same_seed_is_deterministic():
    X, _ = _three_well_separated_clusters()

    config = _fast_config(faiss_seed=47)
    centroids_a, _ = fit_cosine_kmeans(X, n_clusters=3, config=config)
    centroids_b, _ = fit_cosine_kmeans(X, n_clusters=3, config=config)

    np.testing.assert_array_equal(centroids_a, centroids_b)


def test_fit_cosine_kmeans_different_seed_runs_without_error():
    # Not asserted to differ from the seed=47 result (not guaranteed for
    # well-separated data) -- only that a different seed is a valid, usable
    # configuration knob.
    X, _ = _three_well_separated_clusters()
    config = _fast_config(faiss_seed=48)

    centroids, _ = fit_cosine_kmeans(X, n_clusters=3, config=config)
    assert centroids.shape == (3, X.shape[1])


def test_assign_to_centroids_distances_in_valid_range():
    X, _ = _three_well_separated_clusters()
    config = _fast_config()
    centroids, _ = fit_cosine_kmeans(X, n_clusters=3, config=config)

    _, distances = assign_to_centroids(X, centroids)
    assert np.all(distances >= 0.0)
    assert np.all(distances <= 2.0)


def test_assign_to_centroids_point_at_centroid_direction_has_near_zero_distance():
    rng = np.random.default_rng(47)
    centroids = l2_normalize(rng.normal(size=(3, 8)))

    # A point scaled along (but not renormalized to exactly) the first
    # centroid's direction should still be assigned to it at ~0 distance.
    point = (centroids[0] * 5.0).reshape(1, -1)

    labels, distances = assign_to_centroids(point, centroids)
    assert labels[0] == 0
    assert distances[0] < 1e-4


def test_assign_to_centroids_centroids_against_themselves():
    rng = np.random.default_rng(47)
    centroids = l2_normalize(rng.normal(size=(3, 8)))

    labels, distances = assign_to_centroids(centroids, centroids)
    np.testing.assert_array_equal(labels, np.arange(3))
    assert np.all(distances < 1e-4)


def test_l2_normalize_rows_have_unit_norm():
    rng = np.random.default_rng(47)
    X = rng.normal(size=(20, 8))

    X_normalized = l2_normalize(X)
    np.testing.assert_allclose(np.linalg.norm(X_normalized, axis=1), 1.0, atol=1e-8)


def test_l2_normalize_zero_row_does_not_produce_nan_or_inf():
    X = np.zeros((3, 8))
    X[1] = np.array([1.0] + [0.0] * 7)

    X_normalized = l2_normalize(X)
    assert np.all(np.isfinite(X_normalized))
    np.testing.assert_array_equal(X_normalized[0], np.zeros(8))
