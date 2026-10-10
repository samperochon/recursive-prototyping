"""Vocabulary figures: prototype Gram matrix and per-prototype background radii."""

from __future__ import annotations

import numpy as np

from recursive_prototyping.viz.style import (
    LABEL_FONTSIZE,
    MATRIX_KWARGS,
    TITLE_FONTSIZE,
)


def plot_prototype_gram(vocabulary, *, axes=None, split_at=None,
                        title="Prototype similarity matrix"):
    """Cosine-similarity matrix of the vocabulary and a histogram of its off-diagonal entries.

    `split_at` overlays two histograms: the first `split_at` prototypes against each
    other, and all prototypes. Returns (matrix_ax, histogram_ax).
    """
    import matplotlib.pyplot as plt

    vocabulary = np.asarray(vocabulary)
    if vocabulary.ndim != 2 or vocabulary.size == 0:
        raise ValueError(
            f"vocabulary must be a non-empty 2-D array [G, D], got {vocabulary.shape}"
        )
    G = vocabulary.shape[0]
    if split_at is not None and not 0 < split_at < G:
        raise ValueError(f"split_at must be in (0, {G}), got {split_at}")

    normalized = vocabulary / np.maximum(
        np.linalg.norm(vocabulary, axis=1, keepdims=True), 1e-12
    )
    gram = normalized @ normalized.T

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(22, 9))
    ax_matrix, ax_hist = axes

    image = ax_matrix.imshow(gram, **MATRIX_KWARGS)
    ax_matrix.set_title(title, weight="bold", fontsize=TITLE_FONTSIZE)
    ax_matrix.set_xlabel("prototype", fontsize=LABEL_FONTSIZE)
    ax_matrix.set_ylabel("prototype", fontsize=LABEL_FONTSIZE)
    ax_matrix.figure.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04)

    off_diagonal = gram[np.triu_indices(G, k=1)]
    if split_at is None:
        ax_hist.hist(off_diagonal, bins=min(200, max(10, off_diagonal.size)),
                     color="gray", density=True)
    else:
        first = gram[:split_at, :split_at]
        ax_hist.hist(first[np.triu_indices(split_at, k=1)], bins=100, density=True,
                     alpha=0.6, color="tab:blue", label=f"first {split_at} prototypes")
        ax_hist.hist(off_diagonal, bins=100, density=True, alpha=0.6,
                     color="tab:orange", label="all")
        ax_hist.legend(fontsize=LABEL_FONTSIZE)
    ax_hist.set_title(
        f"off-diagonal similarity: mean={off_diagonal.mean():.3f} "
        f"sd={off_diagonal.std():.3f}" if off_diagonal.size
        else "off-diagonal similarity (single prototype)",
        weight="bold", fontsize=TITLE_FONTSIZE,
    )
    ax_hist.set_xlabel("cosine similarity", fontsize=LABEL_FONTSIZE)
    return ax_matrix, ax_hist


def plot_prototype_radii(thresholds, *, ax=None, reference=None, bins=40,
                         title="Per-prototype background radii"):
    """Histogram of the per-prototype radii m_g + n_sigma * sigma_g.

    `thresholds` comes from ``symbolization.assign.compute_prototype_thresholds``;
    `reference` is another such dict to overlay.
    """
    import matplotlib.pyplot as plt

    values = np.asarray(list(dict(thresholds).values()), dtype=float)
    if values.size == 0:
        raise ValueError("thresholds is empty -- nothing to plot")

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    ax.hist(values, bins=bins, color="tab:blue", alpha=0.7,
            label=f"this run (n={values.size})", density=True)
    ax.axvline(float(np.median(values)), color="tab:blue", linestyle="--",
               label=f"median {np.median(values):.3f}")
    if reference is not None:
        ref = np.asarray(list(dict(reference).values()), dtype=float)
        ax.hist(ref, bins=bins, color="tab:orange", alpha=0.5,
                label=f"reference (n={ref.size})", density=True)
        ax.axvline(float(np.median(ref)), color="tab:orange", linestyle="--",
                   label=f"reference median {np.median(ref):.3f}")
    ax.set_title(title, weight="bold", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("radius  m_g + n_sigma * sigma_g  (cosine distance)",
                  fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("density", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LABEL_FONTSIZE)
    return ax
