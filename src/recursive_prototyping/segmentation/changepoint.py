"""ruptures.KernelCPD wrappers for per-participant temporal segmentation.

Change-point arrays follow the convention `[0, t1, ..., T]`: int64, both endpoints
included, strictly increasing. ruptures returns `[t1, ..., T]`; 0 is prepended here.
"""

from __future__ import annotations

import numpy as np
import ruptures

from recursive_prototyping.config.schema import SegmentationConfig

from .kernel import median_heuristic_gamma


def fit_kernel_cpd(
    X: np.ndarray, config: SegmentationConfig, gamma: float | None = None
) -> ruptures.KernelCPD:
    """Fit ruptures.KernelCPD on one participant's sequence `X` [T, D].

    For "rbf", `gamma=None` computes it with `median_heuristic_gamma`; for "linear",
    `gamma` is ignored. `config.jump` is passed through, but KernelCPD always uses 1.
    Raises ValueError if `X` is not 2-D or the kernel is neither "rbf" nor "linear".
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D [T, D], got shape {X.shape}")

    if config.kernel == "rbf":
        if gamma is None:
            gamma = median_heuristic_gamma(X, config.gamma_temperature)
        params = {"gamma": gamma}
    elif config.kernel == "linear":
        params = None
    else:
        raise ValueError(
            f"Unsupported kernel {config.kernel!r}: expected 'rbf' or 'linear'"
        )

    model = ruptures.KernelCPD(
        kernel=config.kernel,
        params=params,
        min_size=config.min_segment_size,
        jump=config.jump,
    )
    return model.fit(X)


def segment_fixed_m(
    X: np.ndarray, n_cpts: int, config: SegmentationConfig, gamma: float | None = None
) -> np.ndarray:
    """Segment `X` at exactly `n_cpts` change points; returns `[0, t1, ..., T]` int64.

    Raises ValueError if `n_cpts < 1`: ruptures rejects 0, whose answer is just `[0, T]`.
    """
    if n_cpts < 1:
        raise ValueError(
            f"n_cpts must be >= 1, got {n_cpts} "
            "(a 0-change-point segmentation is trivially [0, T])"
        )
    model = fit_kernel_cpd(X, config, gamma=gamma)
    bkps = model.predict(n_bkps=n_cpts)
    return np.concatenate(([0], bkps)).astype(np.int64)


def segment_with_penalty(
    X: np.ndarray, penalty: float, config: SegmentationConfig, gamma: float | None = None
) -> np.ndarray:
    """Segment `X` with PELT at a fixed penalty; returns `[0, t1, ..., T]` int64.

    Raises ValueError if `penalty <= 0`: ruptures requires a positive penalty.
    """
    if penalty <= 0:
        raise ValueError(f"penalty must be > 0, got {penalty}")
    model = fit_kernel_cpd(X, config, gamma=gamma)
    bkps = model.predict(pen=penalty)
    return np.concatenate(([0], bkps)).astype(np.int64)
