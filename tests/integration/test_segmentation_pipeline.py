"""Integration tests for segmentation: per-participant kernel
change-point detection (``kernel.py``/``changepoint.py``) driven by the pooled
fixed-effects slope-heuristic penalty calibration (``penalty.py``), exercised
end-to-end via ``calibrate_and_segment`` (docs/METHOD.md, section 2).

Covers, in order:
- the units/convention end-to-end test on hand-built piecewise-constant signals
  with the linear kernel, anchored to the classical ~2*sigma^2-per-change-point
  least-squares result;
- the rbf + per-participant median-heuristic-gamma path on the repo's own
  synthetic dataset, including the repo cpts convention and a margin-based
  recall check against the oracle boundaries;
- Risk(m) monotonicity on real ruptures-backed
  ``collect_risk_curve`` output, for both kernels;
- end-to-end determinism.
"""

from __future__ import annotations

import numpy as np

from recursive_prototyping.config.schema import SegmentationConfig
from recursive_prototyping.segmentation.penalty import calibrate_and_segment
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

# --- TEST 1 setup: hand-built piecewise-constant cohort, linear kernel ---

_NOISE_SIGMA = 0.1  # KNOWN i.i.d. gaussian noise level -- the units anchor is stated in it
_SIGNAL_DIM = 3
_N_TRUE_SEGMENTS = 6  # K_true; every participant has K_true - 1 = 5 true change points
# Three different sequence lengths so the per-participant n_p in lambda_p = -2*n_p*s
# is genuinely exercised: identical lengths would make all penalties coincide.
_LINEAR_LENGTHS = {"p200": 200, "p230": 230, "p260": 260}
# Irregularly spaced true boundaries, as fractions of each participant's length.
_BOUNDARY_FRACS = np.array([0.0, 0.15, 0.35, 0.5, 0.7, 0.85, 1.0])
# Well-separated segment means: consecutive means differ by 3*sqrt(2) ~ 42 sigma.
_SEGMENT_MEANS = 3.0 * np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

_LINEAR_CONFIG = SegmentationConfig(kernel="linear")

# --- TEST 2/3 setup: repo synthetic dataset, rbf kernel ---

_SYNTH_CONFIG = SyntheticDatasetConfig(
    n_participants=3,
    n_action_clusters=3,
    min_windows=60,
    max_windows=100,
    embedding_dim=32,
    seed=47,
)

# Stock defaults, including gamma_temperature=1.0.
_RBF_CONFIG = SegmentationConfig()

_ORACLE_RECALL_MARGIN = 3  # windows; recovered boundaries hit the oracle exactly


def _build_linear_cohort(seed: int = 47) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Hand-build 3 piecewise-constant participants with known true change points.

    Returns (embeddings, true_cpts): per-participant [T, 3] signals with
    _N_TRUE_SEGMENTS well-separated constant segments plus i.i.d. gaussian
    noise of known scale _NOISE_SIGMA, and their repo-convention true
    change-point arrays [0, t1, ..., T].
    """
    rng = np.random.default_rng(seed)
    embeddings: dict[str, np.ndarray] = {}
    true_cpts: dict[str, np.ndarray] = {}
    for participant_id, n_windows in _LINEAR_LENGTHS.items():
        cpts = np.unique(np.round(_BOUNDARY_FRACS * n_windows)).astype(np.int64)
        X = np.zeros((n_windows, _SIGNAL_DIM))
        for k, (start, end) in enumerate(zip(cpts[:-1], cpts[1:])):
            X[start:end] = _SEGMENT_MEANS[k]
        X += rng.normal(scale=_NOISE_SIGMA, size=(n_windows, _SIGNAL_DIM))
        embeddings[participant_id] = X
        true_cpts[participant_id] = cpts
    return embeddings, true_cpts


def _synthetic_rbf_embeddings():
    """The repo's synthetic dataset as calibrate_and_segment input, plus oracles."""
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    embeddings = {pid: p.embeddings for pid, p in participants.items()}
    return participants, embeddings


def _assert_repo_cpts_convention(cpts: np.ndarray, n_windows: int, min_segment_size: int):
    """Repo change-point convention: [0, t1, ..., T] int64, strictly increasing,
    with every segment at least min_segment_size long."""
    assert cpts.dtype == np.int64
    assert cpts[0] == 0
    assert cpts[-1] == n_windows
    assert np.all(np.diff(cpts) > 0)
    assert np.all(np.diff(cpts) >= min_segment_size)


def test_segmentation_units_and_conventions_end_to_end_linear():
    embeddings, true_cpts = _build_linear_cohort()
    result = calibrate_and_segment(embeddings, _LINEAR_CONFIG)

    # (i) Risk(m) decreases in m, so the shared pooled slope is < 0.
    assert result.fit.slope < 0.0
    assert result.fit.n_obs == sum(len(p.m_values) for p in result.participants.values())

    penalties = {}
    for participant_id, part in result.participants.items():
        n_windows = _LINEAR_LENGTHS[participant_id]

        # Risk-curve sanity on the real ruptures-backed collect_risk_curve output:
        # Risk(m) is monotonically non-increasing in m.
        assert np.all(np.diff(part.risks) <= 1e-9)

        # (ii) lambda_p = -2 * n_p * s_hat with each participant's OWN n_p and
        # the shared slope -- an exact identity, which also locks the factor -2
        # being applied exactly once inside the orchestration.
        assert part.penalty > 0.0
        np.testing.assert_allclose(part.penalty, -2.0 * n_windows * result.fit.slope, rtol=1e-12)
        penalties[participant_id] = part.penalty

        # (iii) Non-degenerate recovered change-point count: strictly positive, well
        # below the over-segmentation regime the calibration grid lives in (< the
        # participant's own grid low edge ~ 0.6*m_max = 38-50 here), and in a
        # generous ballpark of the K_true - 1 = 5 true change points. The upper
        # factor is 5x (not 3x): the slope penalty equals twice the AVERAGE
        # marginal risk drop over the deep over-segmentation window
        # [0.6*m_max, m_max], and on white-noise synthetic signals the marginal
        # drop at m ~ 10-20 still exceeds that threshold, so moderate
        # over-segmentation relative to K_true (measured 13-21 under seed 47,
        # deterministic) is a structural property of the calibration grid -- not a
        # units bug, which would move counts to ~0 or toward the grid itself.
        # Exact equality is deliberately NOT asserted (brittle across library
        # versions).
        n_recovered = len(part.cpts) - 2
        n_true = len(true_cpts[participant_id]) - 2
        assert n_true == _N_TRUE_SEGMENTS - 1
        assert 0 < n_recovered < part.m_values[0]
        assert (n_true - 2) <= n_recovered <= 5 * n_true

        _assert_repo_cpts_convention(part.cpts, n_windows, _LINEAR_CONFIG.min_segment_size)

        # (iv) THE UNITS ANCHOR. For the linear kernel the within-segment cost is
        # the classical least-squares cost, where the slope heuristic should land
        # the penalty on the order of the classical 2*sigma^2-per-change-point
        # result (~2*D*sigma^2 = 6*sigma^2 for D=3 dims of i.i.d. noise; measured
        # 5.4-7.0*sigma^2 here). The bracket is deliberately loose -- its job is to
        # catch unit/convention bugs, each of which shifts the penalty by ORDERS of
        # magnitude: a missing n_p factor sends it to ~slope ~ 0.03*sigma^2 (below
        # the bracket); OLS on RAW instead of length-normalized risk inflates the
        # slope ~n_p-fold and sends the penalty ~200x above it. The factor-of-2
        # family (applying the factor 2 twice, see docs/METHOD.md) is locked exactly
        # by the identity in (ii) plus test_penalty_from_slope_applies_factor_of_
        # two_exactly_once and the noiseless slope-recovery unit test.
        sigma2 = _NOISE_SIGMA**2
        assert 0.2 * sigma2 <= part.penalty <= 50.0 * sigma2

    # (ii) continued: participants have pairwise-different n_p, so their penalties
    # must be pairwise different (one shared slope, per-participant rescaling).
    assert len(set(penalties.values())) == len(penalties)


def test_segmentation_rbf_median_heuristic_on_synthetic_dataset():
    participants, embeddings = _synthetic_rbf_embeddings()
    result = calibrate_and_segment(embeddings, _RBF_CONFIG)

    # Per-participant median-heuristic gamma -- finite, positive, and
    # genuinely different from one participant to the next.
    gammas = [part.gamma for part in result.participants.values()]
    assert all(g is not None and np.isfinite(g) and g > 0.0 for g in gammas)
    assert len(set(gammas)) == len(gammas)

    for participant_id, part in result.participants.items():
        n_windows = embeddings[participant_id].shape[0]

        _assert_repo_cpts_convention(part.cpts, n_windows, _RBF_CONFIG.min_segment_size)

        # Non-degenerate: some change points, but nowhere near one every 2 windows.
        n_recovered = len(part.cpts) - 2
        assert 0 < n_recovered < n_windows / 2

        # Risk(m) non-increasing on the rbf path too.
        assert np.all(np.diff(part.risks) <= 1e-9)

        # Loose boundary-quality check: every oracle boundary has a recovered
        # change point within _ORACLE_RECALL_MARGIN windows. Under seed 47 the
        # recovered boundaries contain every oracle boundary EXACTLY;
        # the margin only absorbs library-version drift. (Precision is not
        # asserted: the slope penalty over-segments on this synthetic data, which
        # the count bound above already constrains.)
        oracle_interior = participants[participant_id].oracle_cpts[1:-1]
        recovered_interior = part.cpts[1:-1]
        for boundary in oracle_interior:
            assert np.min(np.abs(recovered_interior - boundary)) <= _ORACLE_RECALL_MARGIN


def test_segmentation_calibrate_and_segment_deterministic():
    # The whole calibrate-then-segment chain is deterministic -- two runs on
    # identical inputs agree bit-for-bit (no RNG anywhere on the segmentation path).
    _, embeddings = _synthetic_rbf_embeddings()
    result_a = calibrate_and_segment(embeddings, _RBF_CONFIG)
    result_b = calibrate_and_segment(embeddings, _RBF_CONFIG)

    assert result_a.fit.slope == result_b.fit.slope
    assert result_a.fit.intercepts == result_b.fit.intercepts
    assert result_a.participants.keys() == result_b.participants.keys()
    for participant_id in result_a.participants:
        part_a = result_a.participants[participant_id]
        part_b = result_b.participants[participant_id]
        assert part_a.gamma == part_b.gamma
        assert part_a.penalty == part_b.penalty
        np.testing.assert_array_equal(part_a.m_values, part_b.m_values)
        np.testing.assert_array_equal(part_a.risks, part_b.risks)
        np.testing.assert_array_equal(part_a.cpts, part_b.cpts)
