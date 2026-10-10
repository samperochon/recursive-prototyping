"""Faiss cosine k-means and nearest-centroid assignment.

Cosine distance is 1 minus the inner product of L2-normalized rows.
"""

from __future__ import annotations

import numpy as np
import faiss

from recursive_prototyping.config.schema import PrototypingConfig

_NORM_EPS = 1e-12


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization.

    Rows with norm below 1e-12 are returned unchanged instead of producing NaN.
    """
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    safe_norms = np.where(norms < _NORM_EPS, 1.0, norms)
    return X / safe_norms


def fit_cosine_kmeans(
    X: np.ndarray, n_clusters: int, config: PrototypingConfig
) -> tuple[np.ndarray, "faiss.Kmeans"]:
    """Cosine k-means via Faiss, fit on L2-normalized `X`.

    Returns (centroids [n_clusters, D], L2-normalized, fitted faiss.Kmeans).
    """
    X_normalized = l2_normalize(X).astype(np.float32)

    kmeans = faiss.Kmeans(
        d=X_normalized.shape[1],
        k=n_clusters,
        niter=config.faiss_niter,
        nredo=config.faiss_nredo,
        spherical=True,
        seed=config.faiss_seed,
        max_points_per_centroid=config.faiss_max_points_per_centroid,
    )
    kmeans.train(X_normalized)

    centroids = np.asarray(kmeans.centroids).reshape(n_clusters, X_normalized.shape[1])
    centroids = l2_normalize(centroids)

    return centroids, kmeans


def assign_to_centroids(
    X: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row of `X` to its nearest centroid under cosine distance.

    Returns (labels [N] int, cosine_distances [N] float = 1 - inner_product,
    in [0, 2]).
    """
    X_normalized = l2_normalize(X).astype(np.float32)
    centroids_normalized = l2_normalize(centroids).astype(np.float32)

    index = faiss.IndexFlatIP(centroids_normalized.shape[1])
    index.add(centroids_normalized)
    inner_products, labels = index.search(X_normalized, 1)

    labels = labels[:, 0].astype(np.int64)
    cosine_distances = 1.0 - inner_products[:, 0].astype(np.float64)

    return labels, cosine_distances
