"""Integration tests for symbolization: nearest-prototype
assignment + background/noise filtering (``assign.py``) feeding the
segment-level central-half majority vote + short-segment smoothing
(``segments.py``), exercised end-to-end (docs/METHOD.md, section 3).

Covers, in order:
- a hand-controlled sequence whose background-threshold arithmetic is
  decisively separated BY CONSTRUCTION, run through the full
  assign -> filter -> symbolize -> expand chain against exact ground truth
  (the decisive correctness test for the background filter);
- the repo's own synthetic dataset with an independently-recomputed oracle
  vocabulary: exact action-segment recovery, repo output conventions, and a
  pinned seed-47 regression anchor for the knife-edge background-segment
  votes (see the marginal-threshold note at TEST 2);
- end-to-end determinism;
- short-segment smoothing repairing an artificially
  over-segmented input back to ground truth.
"""

from __future__ import annotations

import numpy as np

from recursive_prototyping.config.schema import SymbolizationConfig
from recursive_prototyping.symbolization.assign import (
    assign_to_prototypes,
    background_filter,
)
from recursive_prototyping.symbolization.segments import expand_to_windows, symbolize
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

_CONFIG = SymbolizationConfig()  # defaults: background_n_sigma=1.0, min_segment_length=4

# --- TEST 1 setup: hand-controlled sequence, exactly orthonormal vocabulary ---
#
# 3 prototypes = the first 3 standard basis vectors of R^32. Action windows are
# prototype + N(0, 0.05) noise, re-unit-normalized (distance to their own
# prototype ~ 0.02-0.06, measured 0.015-0.055 under seed 47). Background
# windows are unit vectors drawn in the ORTHOGONAL COMPLEMENT of the
# vocabulary span, then tilted by +0.2 toward one chosen prototype per run and
# re-normalized -- so every background window's cosine distance is EXACTLY
# 1 - 0.2/sqrt(1.04) ~ 0.8039 to its tilt target (and 1.0 to the other two),
# independent of the rng draws, and its nearest-prototype assignment is the
# tilt target with a huge inner-product margin (0.196 vs 0.0).
#
# Background-threshold arithmetic, decisive by construction (not seed
# luck): tilting one background run at each prototype gives per-group
# background fractions b_g = 3/32, 4/23, 3/23 ~ 0.09-0.17, so each group's
# distance distribution is two-point-like (action ~ 0.04, background 0.8039)
# with m_g + sigma_g ~ (1-b)*0.04 + b*0.804 + sqrt(b(1-b))*0.76 -- measured
# thresholds 0.333 / 0.460 / 0.395. Background (0.8039) exceeds every
# threshold by >= 0.34 and action (<= 0.055) sits below every threshold by
# >= 0.28: both margins are ~6x the action-distance scale, nothing
# knife-edge. (Naive random-unit-vector background in dim 32 is NOT decisive:
# under seed 47 one raw random window landed at distance 0.4696 vs a 0.4677
# group threshold -- a 0.002 margin -- which is why the background lives in
# the orthogonal complement here.)
_DIM = 32
_VOCABULARY = np.eye(3, _DIM)
_ACTION_NOISE_STD = 0.05
_BACKGROUND_TILT = 0.2
# Ground-truth (symbol, run length): background runs (-1) are short and
# sparse (10/78 windows ~ 13%), every adjacent symbol pair differs, and every
# background run's flanks carry DIFFERENT symbols so short-segment merging is
# a structural no-op on the true segmentation.
_TRUE_SEGMENTS = (
    (0, 10), (1, 8), (-1, 3), (2, 12), (0, 9),
    (-1, 4), (1, 11), (2, 8), (-1, 3), (0, 10),
)

# --- TEST 2 setup: repo synthetic dataset ---

_SYNTH_CONFIG = SyntheticDatasetConfig(
    n_participants=3,
    n_action_clusters=3,
    min_windows=60,
    max_windows=100,
    embedding_dim=32,
    seed=47,
)

# Measured background-segment votes under seed 47, pinned as a regression
# anchor (the chain is fully deterministic) -- see the marginal-threshold
# note in TEST 2 for why these are knife-edge outcomes that cannot be asserted
# categorically: 5 of 7 background segments vote -1; synth_001's two keep
# prototype 1.
_EXPECTED_BACKGROUND_VOTES = {
    "synth_000": [-1, -1],
    "synth_001": [1, 1],
    "synth_002": [-1, -1, -1],
}


def _build_hand_controlled_sequence(
    seed: int = 47,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build TEST 1's sequence from _TRUE_SEGMENTS (see the arithmetic above).

    Returns (X [78, 32] unit rows, per-window truth [78], true_cpts, true
    per-segment symbols).
    """
    rng = np.random.default_rng(seed)
    rows = []
    tilt_target = 0  # background runs tilt toward prototypes 0, 1, 2 in order
    for symbol, length in _TRUE_SEGMENTS:
        if symbol == -1:
            block = rng.normal(size=(length, _DIM))
            block[:, :3] = 0.0  # orthogonal complement of the vocabulary span
            block /= np.linalg.norm(block, axis=1, keepdims=True)
            block = block + _BACKGROUND_TILT * _VOCABULARY[tilt_target]
            tilt_target += 1
        else:
            block = _VOCABULARY[symbol] + rng.normal(
                scale=_ACTION_NOISE_STD, size=(length, _DIM)
            )
        rows.append(block / np.linalg.norm(block, axis=1, keepdims=True))

    true_symbols = np.array([symbol for symbol, _ in _TRUE_SEGMENTS], dtype=np.int64)
    true_cpts = np.concatenate(
        ([0], np.cumsum([length for _, length in _TRUE_SEGMENTS]))
    ).astype(np.int64)
    window_truth = np.repeat(true_symbols, np.diff(true_cpts))
    return np.vstack(rows), window_truth, true_cpts, true_symbols


def _oracle_vocabulary(participants) -> np.ndarray:
    """Per cluster id, the L2-normalized mean of that cluster's windows pooled
    across participants -- an independent-recomputation oracle (the generator
    does not expose its true direction bank)."""
    vocabulary = []
    for g in range(_SYNTH_CONFIG.n_action_clusters):
        pooled = np.vstack(
            [p.embeddings[p.oracle_cluster_ids == g] for p in participants.values()]
        )
        mean = pooled.mean(axis=0)
        vocabulary.append(mean / np.linalg.norm(mean))
    return np.vstack(vocabulary)


def _run_chain(
    X: np.ndarray, vocabulary: np.ndarray, cpts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The full symbolization chain as the pipeline wires it."""
    labels, distances = assign_to_prototypes(X, vocabulary)
    filtered = background_filter(labels, distances, n_sigma=_CONFIG.background_n_sigma)
    final_cpts, final_symbols = symbolize(filtered, cpts, _CONFIG)
    return labels, distances, filtered, final_cpts, final_symbols


def test_symbolization_hand_controlled_end_to_end():
    X, window_truth, true_cpts, true_symbols = _build_hand_controlled_sequence()
    action = window_truth >= 0
    labels, distances, filtered, final_cpts, final_symbols = _run_chain(
        X, _VOCABULARY, true_cpts
    )

    # (i) Raw assignment: every action window labels as its own
    # prototype, and the constructed distance separation is decisive -- action
    # <= 0.055 vs background exactly ~0.8039, on either side of every group
    # threshold (0.333-0.460) with >= 0.28 margin (arithmetic above).
    np.testing.assert_array_equal(labels[action], window_truth[action])
    assert distances[action].max() < 0.15
    assert distances[~action].min() > 0.6

    # (ii) Background filter: exact per-window recovery --
    # every background window relabeled -1, every action window kept.
    np.testing.assert_array_equal(filtered, window_truth)

    # (iii) Vote + smoothing on the TRUE segmentation:
    # per-segment symbols equal the ground-truth construction exactly (action
    # segments -> their prototype index, background segments -> -1), with no
    # spurious merging (background runs have different-symbol flanks).
    np.testing.assert_array_equal(final_cpts, true_cpts)
    np.testing.assert_array_equal(final_symbols, true_symbols)

    # (iv) expand_to_windows round-trips: per-window sequence of the right
    # length, equal to the per-window truth (votes were unanimous).
    windows = expand_to_windows(final_cpts, final_symbols)
    assert len(windows) == len(X) == final_cpts[-1]
    np.testing.assert_array_equal(windows, window_truth)


def test_symbolization_on_synthetic_dataset_with_oracle_vocabulary():
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    vocabulary = _oracle_vocabulary(participants)

    for participant_id, participant in participants.items():
        oracle_cpts = participant.oracle_cpts
        _, _, _, final_cpts, final_symbols = _run_chain(
            participant.embeddings, vocabulary, oracle_cpts
        )

        # (b) Repo output conventions: cpts [0, ..., T] int64 strictly
        # increasing, one int64 symbol per segment, expansion length T.
        assert final_cpts.dtype == np.int64
        assert final_symbols.dtype == np.int64
        assert final_cpts[0] == 0
        assert final_cpts[-1] == len(participant.embeddings)
        assert np.all(np.diff(final_cpts) > 0)
        assert len(final_symbols) == len(final_cpts) - 1
        assert len(expand_to_windows(final_cpts, final_symbols)) == final_cpts[-1]

        # Merging is a structural no-op here: every oracle run is at least
        # _MIN_RUN_LENGTH = 5 >= min_segment_length = 4 windows long except
        # possibly the truncated LAST run, which merging never touches -- so
        # final segments index one-to-one against oracle segments.
        np.testing.assert_array_equal(final_cpts, oracle_cpts)
        segment_truth = participant.oracle_cluster_ids[oracle_cpts[:-1]]

        # (a) Every ACTION segment votes its own oracle cluster id: central
        # halves are unanimous windows at ~0.01-0.07 cosine distance from
        # their own oracle-mean prototype, far under every group threshold.
        action = segment_truth >= 0
        assert action.any()
        np.testing.assert_array_equal(final_symbols[action], segment_truth[action])

        # BACKGROUND segments: with this generator ~50% of windows are
        # background pooled into each prototype's assigned set, so the
        # background threshold m_g + sigma_g lands at
        # ~ f + sqrt(f(1-f)) ~ 1.0 for f ~ 0.5 -- EXACTLY where random-ish
        # background directions sit -- and 'background segment votes -1' is a
        # knife-edge, seed-dependent outcome that cannot be asserted
        # categorically. Instead pin the measured seed-47 votes exactly
        # (deterministic, so non-flaky): synth_001's two background
        # segments keep prototype 1 (its group-1 threshold 0.866 vs
        # background distances from 0.418); the other five vote -1. The
        # decisive background-filter correctness check is TEST 1 above,
        # whose margins are built into the data.
        np.testing.assert_array_equal(
            final_symbols[~action], _EXPECTED_BACKGROUND_VOTES[participant_id]
        )


def test_symbolization_chain_deterministic():
    # The whole assign -> filter -> symbolize chain is deterministic --
    # two runs on identical inputs agree bit-for-bit (no RNG on the symbolization
    # path; SymbolizationConfig.random_seed stays unused).
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    vocabulary = _oracle_vocabulary(participants)
    participant = next(iter(participants.values()))

    result_a = _run_chain(participant.embeddings, vocabulary, participant.oracle_cpts)
    result_b = _run_chain(participant.embeddings, vocabulary, participant.oracle_cpts)
    for array_a, array_b in zip(result_a, result_b):
        np.testing.assert_array_equal(array_a, array_b)


def test_symbolization_short_segment_smoothing_repairs_over_segmentation():
    # Short-segment smoothing exercised end-to-end: split TEST 1's len-12 symbol-2 run
    # [21, 33) with two artificial extra boundaries into [21,27) / [27,29) /
    # [29,33) -- a short (len 2 < 4) middle flanked by same-symbol segments,
    # i.e. over-segmentation inside a homogeneous run. symbolize
    # must merge the three back into one and reproduce the exact ground
    # truth: 12 input segments -> 10 final.
    X, _, true_cpts, true_symbols = _build_hand_controlled_sequence()
    over_cpts = np.sort(np.concatenate((true_cpts, [27, 29]))).astype(np.int64)
    assert 21 in true_cpts and 33 in true_cpts  # the split really is interior to a run

    labels, distances = assign_to_prototypes(X, _VOCABULARY)
    filtered = background_filter(labels, distances, n_sigma=_CONFIG.background_n_sigma)
    final_cpts, final_symbols = symbolize(filtered, over_cpts, _CONFIG)

    assert len(final_symbols) == len(over_cpts) - 1 - 2  # two segments merged away
    np.testing.assert_array_equal(final_cpts, true_cpts)
    np.testing.assert_array_equal(final_symbols, true_symbols)
