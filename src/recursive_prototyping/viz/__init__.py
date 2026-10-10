"""Figures for the recursive-prototyping method.

Each function takes plain arrays and an optional matplotlib ``ax``, returns the ``Axes``
it drew on, and never reads from disk or calls ``plt.show``. ``matplotlib`` is imported
inside each function.
"""

from recursive_prototyping.viz.segmentation import (
    plot_similarity_matrix,
    plot_temperature_ablation,
    plot_trajectory,
)
from recursive_prototyping.viz.symbolic import plot_symbolic_timeline
from recursive_prototyping.viz.vocabulary import (
    plot_prototype_gram,
    plot_prototype_radii,
)

__all__ = [
    "plot_prototype_gram",
    "plot_prototype_radii",
    "plot_similarity_matrix",
    "plot_symbolic_timeline",
    "plot_temperature_ablation",
    "plot_trajectory",
]
