"""Symbolic timeline figure: per-window symbols as a colored strip in time."""

from __future__ import annotations

import numpy as np

from recursive_prototyping.viz.style import (
    LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    symbol_colormap,
)


def plot_symbolic_timeline(window_symbols, *, ax=None, hop_seconds=None,
                           symbol_names=None, title="Symbolic timeline",
                           colorbar=True):
    """One participant's per-window symbols as a colored time strip.

    `window_symbols` is the output of ``expand_to_windows``; -1 is drawn gray. A 2-D
    array stacks several rows, which share one colormap and so must use the same
    alphabet. `hop_seconds` relabels the x axis in minutes (0.32 s for the paper; see
    ``penalty.DEFAULT_HOP_SECONDS``). `symbol_names` maps symbol -> colorbar label.
    """
    import matplotlib.pyplot as plt

    symbols = np.asarray(window_symbols)
    if symbols.ndim == 1:
        symbols = symbols[None, :]
    if symbols.ndim != 2 or symbols.size == 0:
        raise ValueError(
            f"window_symbols must be a non-empty 1-D or 2-D array, got {symbols.shape}"
        )

    cmap, norm, ordered = symbol_colormap(symbols)
    # Rank of each symbol, so the colormap indexes densely.
    ranks = np.searchsorted(ordered, symbols)

    if ax is None:
        _, ax = plt.subplots(figsize=(25, 1.2 * symbols.shape[0] + 1.5))
    extent = (0, symbols.shape[1], symbols.shape[0], 0)
    image = ax.imshow(ranks, aspect="auto", interpolation="nearest", cmap=cmap,
                      norm=norm, extent=extent)
    ax.set_yticks(np.arange(symbols.shape[0]) + 0.5)
    ax.set_yticklabels([f"row {i}" for i in range(symbols.shape[0])])
    ax.set_title(title, weight="bold", fontsize=TITLE_FONTSIZE)

    if hop_seconds is None:
        ax.set_xlabel("embedding index", fontsize=LABEL_FONTSIZE)
    else:
        n_windows = symbols.shape[1]
        ticks = np.linspace(0, n_windows, min(11, n_windows + 1))
        ax.set_xticks(ticks)
        if n_windows * hop_seconds < 120:
            ax.set_xticklabels([f"{t * hop_seconds:.0f}" for t in ticks])
            ax.set_xlabel("time (s)", fontsize=LABEL_FONTSIZE)
        else:
            ax.set_xticklabels([f"{t * hop_seconds / 60:.0f}" for t in ticks])
            ax.set_xlabel("time (minutes)", fontsize=LABEL_FONTSIZE)

    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, ticks=np.arange(len(ordered)),
                                 fraction=0.02, pad=0.01)
        names = dict(symbol_names or {})
        bar.ax.set_yticklabels(
            [names.get(int(s), "background" if s == -1 else str(int(s)))
             for s in ordered]
        )
    return ax
