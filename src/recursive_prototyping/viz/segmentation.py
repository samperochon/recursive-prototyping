"""Segmentation figures: embedding trajectory, self-similarity, temperature ablation."""

from __future__ import annotations

import numpy as np

from recursive_prototyping.viz.style import (
    LABEL_FONTSIZE,
    MATRIX_KWARGS,
    TITLE_FONTSIZE,
    draw_change_points,
)


def _window(X: np.ndarray, window: tuple[int, int] | None) -> tuple[np.ndarray, int]:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D [T, D], got shape {X.shape}")
    if window is None:
        return X, 0
    start, end = window
    if not 0 <= start < end <= X.shape[0]:
        raise ValueError(
            f"window {window} out of range for T={X.shape[0]} "
            "(need 0 <= start < end <= T)"
        )
    return X[start:end], start


def plot_trajectory(X, cpts=None, *, window=None, ax=None,
                    title="Embedding trajectory with change points"):
    """Heatmap of the embedding sequence `X` [T, D], time along x, with change points.

    `window=(start, end)` restricts the panel to `X[start:end]`.
    """
    import matplotlib.pyplot as plt

    Xw, offset = _window(X, window)
    if ax is None:
        _, ax = plt.subplots(figsize=(25, 8))
    ax.imshow(Xw.T, **MATRIX_KWARGS)
    if cpts is not None:
        draw_change_points(ax, cpts, vertical=True, offset=offset)
    ax.set_title(title, weight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("embedding index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("feature", fontsize=LABEL_FONTSIZE)
    return ax


def plot_similarity_matrix(X, cpts=None, *, window=None, ax=None, normalize=True,
                           title="Self-similarity with change points"):
    """Self-similarity matrix of an embedding sequence with change points overlaid.

    `normalize=True` L2-normalizes rows first (cosine similarity); False shows raw inner
    products. Change points are drawn both vertically and horizontally.
    """
    import matplotlib.pyplot as plt

    Xw, offset = _window(X, window)
    if normalize:
        norms = np.linalg.norm(Xw, axis=1, keepdims=True)
        Xw = Xw / np.maximum(norms, 1e-12)
    gram = Xw @ Xw.T

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(gram, **MATRIX_KWARGS)
    if cpts is not None:
        draw_change_points(ax, cpts, vertical=True, horizontal=True, offset=offset)
    ax.set_title(title, weight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("embedding index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("embedding index", fontsize=LABEL_FONTSIZE)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_temperature_ablation(X, temperatures=(0.25, 0.5, 1.0, 2.0, 4.0), *,
                              window=None, axes=None):
    """RBF kernel matrix and its off-diagonal histogram, one row per temperature.

    The kernel is `exp(-gamma ||x - y||^2)` with
    `gamma = median_heuristic_gamma(X, temperature)`; a larger temperature gives a
    smaller gamma and a flatter kernel. Returns the (n_temperatures, 2) array of Axes.
    """
    import matplotlib.pyplot as plt

    from recursive_prototyping.segmentation.kernel import median_heuristic_gamma

    Xw, _ = _window(X, window)
    temperatures = list(temperatures)
    if not temperatures:
        raise ValueError("temperatures must be non-empty")

    if axes is None:
        _, axes = plt.subplots(
            len(temperatures), 2, figsize=(16, 4 * len(temperatures)), squeeze=False
        )
    axes = np.asarray(axes).reshape(len(temperatures), 2)

    sq_dists = ((Xw[:, None, :] - Xw[None, :, :]) ** 2).sum(-1)
    upper = np.triu_indices(Xw.shape[0], k=1)
    for row, temperature in enumerate(temperatures):
        gamma = median_heuristic_gamma(Xw, temperature)
        kernel = np.exp(-gamma * sq_dists)
        values = kernel[upper]

        ax_matrix, ax_hist = axes[row]
        ax_matrix.imshow(kernel, **MATRIX_KWARGS)
        ax_matrix.set_title(
            f"T={temperature:g}  gamma={gamma:.3g}", weight="bold",
            fontsize=TITLE_FONTSIZE,
        )
        ax_hist.hist(values, bins=50, color="gray")
        ax_hist.set_title(
            f"off-diagonal: mean={values.mean():.3f} sd={values.std():.3f}",
            weight="bold", fontsize=TITLE_FONTSIZE,
        )
        ax_hist.set_xlabel("kernel value", fontsize=LABEL_FONTSIZE)
    return axes
