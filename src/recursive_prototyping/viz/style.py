"""Shared figure style.

Applied per call rather than through ``rcParams``, so importing this module does not
change global matplotlib state.
"""

from __future__ import annotations

import numpy as np

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 14
CPT_COLOR = "cyan"
CPT_LINEWIDTH = 2.0
BACKGROUND_COLOR = "#bdbdbd"
"""Color of the background symbol -1."""

MATRIX_KWARGS = {"cmap": "coolwarm", "interpolation": "nearest", "aspect": "auto"}
"""imshow defaults for matrix panels."""


def fi(x: float = 25, y: float = 4):
    """Shorthand for ``plt.figure(figsize=(x, y))``."""
    import matplotlib.pyplot as plt

    return plt.figure(figsize=(x, y))


def cluster_colors(n: int) -> list[tuple[float, float, float, float]]:
    """``n`` colors from the tab20 + tab20b + tab20c cycle; they repeat beyond 60."""
    import matplotlib.pyplot as plt

    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    palette = [
        color
        for name in ("tab20", "tab20b", "tab20c")
        for color in plt.get_cmap(name).colors
    ]
    return [palette[i % len(palette)] for i in range(n)]


def symbol_colormap(symbols: np.ndarray):
    """Return (cmap, norm, ordered_symbols) for a symbol array that may contain -1.

    -1 maps to ``BACKGROUND_COLOR``; other symbols take ``cluster_colors`` in order.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ordered = np.unique(np.asarray(symbols))
    non_background = [s for s in ordered if s != -1]
    colors = []
    if -1 in ordered:
        colors.append(BACKGROUND_COLOR)
    colors.extend(cluster_colors(len(non_background)))
    cmap = ListedColormap(colors)
    boundaries = np.arange(len(ordered) + 1) - 0.5
    return cmap, BoundaryNorm(boundaries, cmap.N), ordered


def draw_change_points(ax, cpts, *, vertical: bool = True, horizontal: bool = False,
                       offset: int = 0) -> None:
    """Overlay the interior change points of ``cpts`` (``[0, t1, ..., T]``) as dashed lines.

    ``offset`` is subtracted from each position, for panels showing ``X[a:b]``.
    """
    cpts = np.asarray(cpts)
    interior = cpts[1:-1] if cpts.size >= 2 else cpts
    for t in interior:
        position = int(t) - offset
        if vertical:
            ax.axvline(position, color=CPT_COLOR, linestyle="--", lw=CPT_LINEWIDTH)
        if horizontal:
            ax.axhline(position, color=CPT_COLOR, linestyle="--", lw=CPT_LINEWIDTH)
