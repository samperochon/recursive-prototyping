"""Window-to-prototype assignment and background filtering (paper Section 3.3).

Each window is assigned to its nearest accepted centroid under cosine distance, and the
centroid is mapped to its prototype. Windows unusually far from their centroid are then
relabeled as background (-1).
"""

from __future__ import annotations

import numpy as np

from recursive_prototyping.prototyping.clustering import assign_to_centroids


def assign_to_prototypes(
    X: np.ndarray,
    centroids: np.ndarray,
    centroid_to_prototype: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each window to a prototype through its nearest centroid (paper Section 3.3).

    y_i = Phi(argmin_k d(x_i, mu_k)): each row of `X` [N, D] finds its nearest row of
    `centroids` [K, D] under cosine distance, and `centroid_to_prototype` [K] maps that
    centroid index to a prototype label (identity if omitted).

    Returns (labels [N] int64, cosine distances [N] float64 to the nearest centroid).
    Raises ValueError on mismatched shapes.
    """
    X = np.asarray(X)
    centroids = np.asarray(centroids)
    if centroids.ndim != 2 or centroids.size == 0:
        raise ValueError(
            f"centroids must be a non-empty 2-D array [K, D], got shape {centroids.shape}"
        )
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array [N, D], got shape {X.shape}")
    if X.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"X and centroids embedding dims differ: {X.shape[1]} vs {centroids.shape[1]}"
        )
    labels, distances = assign_to_centroids(X, centroids)
    if centroid_to_prototype is None:
        return labels, distances
    centroid_to_prototype = np.asarray(centroid_to_prototype)
    if centroid_to_prototype.shape != (centroids.shape[0],):
        raise ValueError(
            f"centroid_to_prototype must have one entry per centroid ({centroids.shape[0]}), "
            f"got shape {centroid_to_prototype.shape}"
        )
    return centroid_to_prototype[labels].astype(np.int64), distances


def _check_raw_labels(labels: np.ndarray) -> None:
    """Reject an already-filtered label array (its -1s would form a fake group)."""
    if labels.size > 0 and labels.min() < 0:
        raise ValueError(
            "labels must contain only prototype indices >= 0 (raw assignment "
            "output); found a negative label -- was background_filter already applied?"
        )


def compute_prototype_thresholds(
    labels: np.ndarray, distances: np.ndarray, n_sigma: float = 1.0
) -> dict[int, float]:
    """Per-prototype radius m_g + n_sigma * sigma_g, as ``{g: radius}`` (paper Section 3.3).

    m_g and sigma_g are the mean and population std of `distances` over the windows
    labeled g. The radii are meant to be pooled over the whole cohort, which is the
    caller's job. Raises ValueError if `labels` contains a negative value.
    """
    labels = np.asarray(labels)
    distances = np.asarray(distances)
    _check_raw_labels(labels)
    thresholds: dict[int, float] = {}
    for g in np.unique(labels):
        group_distances = distances[labels == g]
        thresholds[int(g)] = float(
            group_distances.mean() + n_sigma * np.std(group_distances)
        )
    return thresholds


def background_filter(
    labels: np.ndarray,
    distances: np.ndarray,
    n_sigma: float = 1.0,
    thresholds: dict[int, float] | None = None,
) -> np.ndarray:
    """Relabel windows with d > m_g + n_sigma * sigma_g as background (-1), in a new array.

    Pass ``thresholds`` (from ``compute_prototype_thresholds``) to apply cohort-pooled
    radii to one participant; ``n_sigma`` is then ignored. The inequality is strict, so
    a prototype with a single window is never filtered.

    Raises ValueError if `labels` is already filtered (contains a negative value) and
    KeyError if ``thresholds`` lacks a prototype present in `labels`.
    """
    labels = np.asarray(labels)
    distances = np.asarray(distances)
    _check_raw_labels(labels)

    if thresholds is None:
        thresholds = compute_prototype_thresholds(labels, distances, n_sigma)

    filtered = labels.astype(np.int64)  # astype copies by default
    for g in np.unique(labels):
        key = int(g)
        if key not in thresholds:
            raise KeyError(
                f"no threshold supplied for prototype {key}; "
                "compute_prototype_thresholds must cover every label present"
            )
        filtered[(labels == g) & (distances > thresholds[key])] = -1
    return filtered
