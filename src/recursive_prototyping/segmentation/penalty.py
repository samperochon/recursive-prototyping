"""Slope-heuristic penalty calibration for kernel change-point detection.

One pooled OLS regression of length-normalized risk on the number of change points
(per-participant intercepts, one shared slope) gives the slope; each participant's
penalty is -2 * n_windows * slope. Sibling modules are imported inside functions so
this module imports without ruptures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from recursive_prototyping.config.schema import SegmentationConfig


DEFAULT_HOP_SECONDS = 0.32
"""Seconds of video per embedding index: stride 8 at 25 fps.

Not the 0.64 s window length: windows of 16 frames overlap by half.
"""


def default_m_max(
    n_windows: int,
    config: SegmentationConfig,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> int:
    """Top of the calibration grid: duration_seconds / ``seconds_per_segment_default``.

    ``config.m_max`` overrides it. The result is clipped to the largest count ruptures
    can fit, n_windows // min_segment_size - 1, then raised to at least 2; for sequences
    shorter than 3 * min_segment_size it is therefore infeasible and the fit fails.
    """
    if config.m_max is not None:
        m_max = config.m_max
    else:
        duration_seconds = n_windows * hop_seconds
        m_max = int(round(duration_seconds / config.seconds_per_segment_default))
    feasible_max = n_windows // config.min_segment_size - 1
    return max(min(m_max, feasible_max), 2)


def m_grid(m_max: int, config: SegmentationConfig) -> np.ndarray:
    """Ascending unique int64 change-point counts (>= 1) in [m_grid_low_frac * m_max, m_max].

    Rounding removes duplicates, so fewer than ``m_grid_size`` values may come back.
    """
    low = config.m_grid_low_frac * m_max
    grid = np.unique(np.round(np.linspace(low, m_max, config.m_grid_size)))
    grid = grid.astype(np.int64)
    return grid[grid >= 1]


def collect_risk_curve(
    X: np.ndarray,
    m_values: np.ndarray,
    config: SegmentationConfig,
    gamma: float | None = None,
) -> np.ndarray:
    """Unpenalized risk (raw cost sum) for every m in ``m_values``, from one KernelCPD fit.

    Counts are visited largest first so ruptures reuses its dynamic-programming path;
    risks come back aligned with ``m_values``.
    """
    from recursive_prototyping.segmentation.changepoint import fit_kernel_cpd

    model = fit_kernel_cpd(X, config, gamma)
    m_values = np.asarray(m_values)
    risks = np.empty(m_values.shape[0], dtype=np.float64)
    for idx in np.argsort(m_values)[::-1]:
        bkps = model.predict(n_bkps=int(m_values[idx]))
        risks[idx] = model.cost.sum_of_costs(bkps)
    return risks


def pool_risk_curves(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, int]],
) -> pd.DataFrame:
    """Stack per-participant (m_values, risks, n_windows) curves into one long table,
    adding normalized_risk = risk / n_windows."""
    frames = [
        pd.DataFrame(
            {
                "participant_id": participant_id,
                "m": np.asarray(m_values, dtype=np.int64),
                "risk": np.asarray(risks, dtype=np.float64),
                "n_windows": np.int64(n_windows),
            }
        )
        for participant_id, (m_values, risks, n_windows) in curves.items()
    ]
    pooled = pd.concat(frames, ignore_index=True)
    pooled["normalized_risk"] = pooled["risk"] / pooled["n_windows"]
    return pooled


@dataclass
class PooledSlopeFit:
    """Fitted pooled fixed-effects regression normalized_risk = a_p + s * m."""

    slope: float
    """Shared slope: the fitted OLS coefficient on m, with no factor applied."""

    intercepts: dict[str, float]
    """One intercept per participant_id."""

    r_squared: float
    """Centered R^2 of the pooled fit."""

    n_obs: int
    """Number of pooled (participant, m) observations."""


def estimate_pooled_slope(pooled: pd.DataFrame) -> PooledSlopeFit:
    """Pooled OLS of normalized risk on m: per-participant intercepts, one shared slope.

    One dummy column per participant and no global constant; the dummies span it, hence
    ``hasconst=True`` so the R^2 is centered.
    """
    dummies = pd.get_dummies(pooled["participant_id"], dtype=np.float64)
    design = pd.concat([dummies, pooled["m"].astype(np.float64)], axis=1)
    endog = pooled["normalized_risk"].astype(np.float64)
    result = sm.OLS(endog, design, hasconst=True).fit()
    return PooledSlopeFit(
        slope=float(result.params["m"]),
        intercepts={pid: float(result.params[pid]) for pid in dummies.columns},
        r_squared=float(result.rsquared),
        n_obs=int(result.nobs),
    )


def penalty_from_slope(slope: float, n_windows: int) -> float:
    """Penalty = -2 * n_windows * slope, the factor -2 being applied once, here.

    ``n_windows`` is the participant's own sequence length; it converts the normalized
    slope back to ruptures' raw-cost units.
    """
    return -2.0 * n_windows * slope


@dataclass
class ParticipantCalibration:
    """Per-participant record produced by ``calibrate_and_segment``."""

    gamma: float | None
    """Median-heuristic RBF gamma; None for the linear kernel."""

    m_max: int
    """Top of this participant's calibration grid."""

    m_values: np.ndarray
    """Candidate change-point counts."""

    risks: np.ndarray
    """Unpenalized risk for each entry of ``m_values``."""

    penalty: float
    """-2 * n_windows * slope."""

    cpts: np.ndarray
    """Penalized segmentation, [0, t1, ..., T]."""


@dataclass
class CalibrationResult:
    """Output of ``calibrate_and_segment``: the shared fit and per-participant records."""

    fit: PooledSlopeFit
    participants: dict[str, ParticipantCalibration]


def calibrate_and_segment(
    embeddings: Mapping[str, np.ndarray],
    config: SegmentationConfig,
    hop_seconds: float | Mapping[str, float] = DEFAULT_HOP_SECONDS,
) -> CalibrationResult:
    """Calibrate the penalty over a cohort and segment each participant.

    Per participant: gamma (rbf only), the m grid and its risk curve. One pooled OLS
    gives the shared slope; each participant is then segmented with PELT at its own
    penalty. ``hop_seconds`` is one value or a mapping participant_id -> seconds per
    embedding index.
    """
    from recursive_prototyping.segmentation.changepoint import segment_with_penalty
    from recursive_prototyping.segmentation.kernel import median_heuristic_gamma

    gammas: dict[str, float | None] = {}
    m_maxes: dict[str, int] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for participant_id, X in embeddings.items():
        n_windows = X.shape[0]
        gamma = (
            median_heuristic_gamma(X, config.gamma_temperature)
            if config.kernel == "rbf"
            else None
        )
        hop_p = (
            hop_seconds[participant_id]
            if isinstance(hop_seconds, Mapping)
            else hop_seconds
        )
        m_max_p = default_m_max(n_windows, config, hop_p)
        grid = m_grid(m_max_p, config)
        risks = collect_risk_curve(X, grid, config, gamma)
        gammas[participant_id] = gamma
        m_maxes[participant_id] = m_max_p
        curves[participant_id] = (grid, risks, n_windows)

    fit = estimate_pooled_slope(pool_risk_curves(curves))

    participants: dict[str, ParticipantCalibration] = {}
    for participant_id, X in embeddings.items():
        n_windows = X.shape[0]
        penalty = penalty_from_slope(fit.slope, n_windows)
        cpts = segment_with_penalty(X, penalty, config, gamma=gammas[participant_id])
        participants[participant_id] = ParticipantCalibration(
            gamma=gammas[participant_id],
            m_max=m_maxes[participant_id],
            m_values=curves[participant_id][0],
            risks=curves[participant_id][1],
            penalty=penalty,
            cpts=cpts,
        )
    return CalibrationResult(fit=fit, participants=participants)
