"""RBF kernel bandwidth by the median heuristic: gamma = 1 / (2 * temperature * sigma^2).

sigma is the median pairwise Euclidean distance of one participant's sequence. The
temperature divides, so raising it lowers gamma (a wider kernel).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def median_heuristic_gamma(X: np.ndarray, temperature: float = 1.0) -> float:
    """Return gamma = 1 / (2 * temperature * sigma^2) for one sequence ``X`` [T, D].

    Raises ValueError if ``X`` is not 2-D, has fewer than 2 rows, ``temperature <= 0``,
    or the median pairwise distance is 0.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array [T, D], got ndim={X.ndim}")

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError(
            "X must have at least 2 rows to compute pairwise distances, "
            f"got T={n_samples}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    sigma_rbf = float(np.median(np.sqrt(pdist(X, "sqeuclidean"))))
    if sigma_rbf == 0.0:
        raise ValueError(
            "sigma_rbf is 0 (median pairwise distance is zero, e.g. all rows "
            "of X are identical) -- gamma is undefined for a degenerate "
            "sequence"
        )

    return float(1.0 / (2.0 * temperature * sigma_rbf**2))
