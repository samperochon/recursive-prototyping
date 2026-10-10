"""Unit tests for recursive_prototyping.viz.

These assert the things a figure can actually get wrong -- shapes, the
number of overlaid lines, which colors a symbol maps to, input validation
-- rather than pixel output. Every test runs on the Agg backend and closes
its figures, so the suite stays headless and leak-free.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from recursive_prototyping.viz.segmentation import (  # noqa: E402
    plot_similarity_matrix,
    plot_temperature_ablation,
    plot_trajectory,
)
from recursive_prototyping.viz.style import (  # noqa: E402
    BACKGROUND_COLOR,
    cluster_colors,
    draw_change_points,
    symbol_colormap,
)
from recursive_prototyping.viz.symbolic import plot_symbolic_timeline  # noqa: E402
from recursive_prototyping.viz.vocabulary import (  # noqa: E402
    plot_prototype_gram,
    plot_prototype_radii,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _piecewise(n_windows=60, dim=8, seed=47):
    """A 3-segment piecewise-constant sequence with known boundaries."""
    rng = np.random.default_rng(seed)
    cpts = np.array([0, 20, 40, n_windows])
    X = np.zeros((n_windows, dim))
    for k, (start, end) in enumerate(zip(cpts[:-1], cpts[1:])):
        X[start:end] = rng.normal(size=dim) * 3
    return X + rng.normal(scale=0.1, size=(n_windows, dim)), cpts


# --------------------------------------------------------------- style ---
def test_cluster_colors_cycles_beyond_the_palette():
    colors = cluster_colors(65)
    assert len(colors) == 65
    # 60 distinct colors in tab20+tab20b+tab20c, then wraparound.
    assert colors[60] == colors[0]
    assert len(set(colors[:60])) == 60


def test_cluster_colors_rejects_negative():
    with pytest.raises(ValueError, match="n must be >= 0"):
        cluster_colors(-1)


def test_symbol_colormap_puts_background_first_and_gray():
    cmap, _, ordered = symbol_colormap(np.array([-1, 0, 2, 5]))
    np.testing.assert_array_equal(ordered, [-1, 0, 2, 5])
    assert cmap.N == 4
    assert matplotlib.colors.to_hex(cmap.colors[0]) == BACKGROUND_COLOR


def test_symbol_colormap_without_background_uses_no_gray_slot():
    cmap, _, ordered = symbol_colormap(np.array([3, 4]))
    np.testing.assert_array_equal(ordered, [3, 4])
    assert cmap.N == 2
    assert matplotlib.colors.to_hex(cmap.colors[0]) != BACKGROUND_COLOR


def test_draw_change_points_drops_both_endpoints():
    _, ax = plt.subplots()
    draw_change_points(ax, np.array([0, 10, 20, 30]))
    # [0, 10, 20, 30] -> interior 10, 20 only.
    xs = sorted(line.get_xdata()[0] for line in ax.lines)
    assert xs == [10, 20]


def test_draw_change_points_applies_offset():
    _, ax = plt.subplots()
    draw_change_points(ax, np.array([0, 2100, 2500, 3000]), offset=2000)
    xs = sorted(line.get_xdata()[0] for line in ax.lines)
    assert xs == [100, 500]


# -------------------------------------------------------- segmentation ---
def test_plot_trajectory_transposes_so_time_is_the_x_axis():
    X, cpts = _piecewise()
    ax = plot_trajectory(X, cpts)
    image = ax.images[0].get_array()
    assert image.shape == (X.shape[1], X.shape[0])  # [D, T]
    assert len(ax.lines) == len(cpts) - 2


def test_plot_similarity_matrix_is_square_and_normalized():
    X, cpts = _piecewise()
    ax = plot_similarity_matrix(X, cpts)
    gram = np.asarray(ax.images[0].get_array())
    assert gram.shape == (X.shape[0], X.shape[0])
    # Normalized rows => unit diagonal, everything within [-1, 1].
    np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-6)
    assert gram.min() >= -1.0001 and gram.max() <= 1.0001
    # Both vertical and horizontal lines per interior change point.
    assert len(ax.lines) == 2 * (len(cpts) - 2)


def test_plot_similarity_matrix_window_restricts_and_shifts_cpts():
    X, _ = _piecewise(n_windows=100)
    ax = plot_similarity_matrix(X, np.array([0, 30, 60, 100]), window=(20, 70))
    gram = np.asarray(ax.images[0].get_array())
    assert gram.shape == (50, 50)
    # axvline has constant xdata; axhline does not -- keep only the verticals.
    xs = sorted(
        line.get_xdata()[0] for line in ax.lines
        if line.get_xdata()[0] == line.get_xdata()[1]
    )
    assert xs == [10, 40]  # 30 - 20 and 60 - 20


def test_plot_similarity_matrix_rejects_out_of_range_window():
    X, _ = _piecewise(n_windows=50)
    with pytest.raises(ValueError, match="out of range"):
        plot_similarity_matrix(X, window=(10, 500))


def test_plot_trajectory_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        plot_trajectory(np.arange(10))


def test_temperature_ablation_shape_and_monotone_gamma():
    X, _ = _piecewise(n_windows=30)
    temperatures = (0.5, 1.0, 2.0)
    axes = plot_temperature_ablation(X, temperatures)
    assert axes.shape == (3, 2)
    # The temperature DIVIDES gamma, so the kernel widens (mean rises)
    # monotonically as temperature grows.
    means = []
    for row in range(3):
        kernel = np.asarray(axes[row, 0].images[0].get_array())
        means.append(kernel[np.triu_indices(kernel.shape[0], k=1)].mean())
    assert means[0] < means[1] < means[2]


def test_temperature_ablation_rejects_empty_temperatures():
    X, _ = _piecewise(n_windows=20)
    with pytest.raises(ValueError, match="non-empty"):
        plot_temperature_ablation(X, ())


# ---------------------------------------------------------- vocabulary ---
def test_plot_prototype_gram_panels_and_unit_diagonal():
    rng = np.random.default_rng(47)
    vocabulary = rng.normal(size=(6, 12))
    ax_matrix, ax_hist = plot_prototype_gram(vocabulary)
    gram = np.asarray(ax_matrix.images[0].get_array())
    assert gram.shape == (6, 6)
    np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-6)
    assert len(ax_hist.patches) > 0


def test_plot_prototype_gram_split_at_adds_a_second_population():
    rng = np.random.default_rng(47)
    vocabulary = rng.normal(size=(8, 5))
    _, ax_hist = plot_prototype_gram(vocabulary, split_at=3)
    labels = [t.get_text() for t in ax_hist.get_legend().get_texts()]
    assert labels == ["first 3 prototypes", "all"]


def test_plot_prototype_gram_rejects_bad_split_at():
    rng = np.random.default_rng(47)
    with pytest.raises(ValueError, match="split_at"):
        plot_prototype_gram(rng.normal(size=(4, 3)), split_at=4)


def test_plot_prototype_gram_rejects_empty_vocabulary():
    with pytest.raises(ValueError, match="non-empty"):
        plot_prototype_gram(np.zeros((0, 4)))


def test_plot_prototype_radii_overlays_reference():
    thresholds = {0: 0.4, 1: 0.5, 2: 0.45}
    reference = {0: 0.30, 1: 0.35}
    ax = plot_prototype_radii(thresholds, reference=reference)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "this run (n=3)" in labels
    assert "reference (n=2)" in labels
    # Median guide lines for both populations.
    assert len(ax.lines) == 2


def test_plot_prototype_radii_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        plot_prototype_radii({})


# ------------------------------------------------------------ symbolic ---
def test_symbolic_timeline_ranks_noncontiguous_symbols_densely():
    # Symbols 0 and 7 with background: the colormap has 3 slots, and the
    # drawn array must use ranks 0..2, not the raw values.
    symbols = np.array([-1, -1, 0, 0, 7, 7])
    ax = plot_symbolic_timeline(symbols)
    drawn = np.asarray(ax.images[0].get_array())
    assert drawn.shape == (1, 6)
    np.testing.assert_array_equal(drawn[0], [0, 0, 1, 1, 2, 2])


def test_symbolic_timeline_stacks_rows():
    symbols = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
    ax = plot_symbolic_timeline(symbols)
    assert np.asarray(ax.images[0].get_array()).shape == (2, 4)


def test_symbolic_timeline_hop_seconds_labels_axis_in_minutes():
    symbols = np.zeros(600, dtype=np.int64)
    ax = plot_symbolic_timeline(symbols, hop_seconds=0.32)
    assert ax.get_xlabel() == "time (minutes)"
    # 600 windows * 0.32 s = 192 s = 3.2 min -> last tick label "3".
    assert ax.get_xticklabels()[-1].get_text() == "3"


def test_symbolic_timeline_names_background_in_the_colorbar():
    ax = plot_symbolic_timeline(np.array([-1, 0, 1]), symbol_names={0: "stir", 1: "pour"})
    bar_labels = [t.get_text() for t in ax.figure.axes[-1].get_yticklabels()]
    assert bar_labels == ["background", "stir", "pour"]


def test_symbolic_timeline_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        plot_symbolic_timeline(np.array([]))
