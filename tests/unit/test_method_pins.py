"""Hand-computed pins for method rules that a plausible wrong implementation would still
pass elsewhere in the suite. Each test fails on a one-line change of the rule that the
rest of the suite does not catch."""

from pathlib import Path

import numpy as np
import pytest

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.consolidation import (
    ConsolidationResult,
    MembershipTree,
    assemble_final_vocabulary,
)
from recursive_prototyping.prototyping.refinement import (
    PendingRound,
    build_residual_set,
    finish_round,
    init_refinement_state,
    start_round,
)
from recursive_prototyping.symbolization.assign import compute_prototype_thresholds
from recursive_prototyping.synthetic.frames import SyntheticFrameSource
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

# --- residual set --------------------------------------------------------------------------


def test_residual_distance_equal_to_d_min_is_not_recycled():
    # The paper recycles accepted-cluster samples "above" d_min: strictly greater.
    mask = build_residual_set(
        np.array([0, 0]), np.array([0.3, 0.30000001]), accepted_ids={0}, ambiguous_ids=set(),
        d_min=0.3,
    )
    np.testing.assert_array_equal(mask, [False, True])


def _pending_round(output_dir):
    """10 global rows; this round's R_p is rows {2, 5, 7, 8} (the others were dropped
    earlier); clusters 0 accepted, 1 ambiguous, 2 noise."""
    probe_paths = [output_dir / "a.png", output_dir / "b.png", output_dir / "c.png"]
    pending = PendingRound(
        round=1,
        residual_global_indices=np.array([2, 5, 7, 8]),
        centroids=np.eye(3, 4, dtype=np.float32),
        labels=np.array([0, 1, 0, 2]),
        distances=np.array([0.5, 0.1, 0.3, 0.9]),  # row 2: 0.5 > 0.3 kept; row 7: == 0.3 not
        probe_paths=probe_paths,
    )
    return pending, probe_paths


def test_finish_round_residual_is_the_hand_computed_subset_of_the_pending_round(output_dir):
    X_train = np.eye(10, 4, dtype=np.float32)
    state = init_refinement_state(X_train)
    state.residual_mask[:] = False
    state.residual_mask[[2, 5, 7, 8]] = True
    state.pending, probe_paths = _pending_round(output_dir)
    annotations = [
        (probe_paths[0], "task-definitive"),
        (probe_paths[1], "exo-ambiguous"),
        (probe_paths[2], "noise"),
    ]

    finish_round(state, annotations, PrototypingConfig(residual_distance_threshold=0.3),
                 "alice", output_dir)

    # ambiguous row 5 kept; accepted row 2 (d=0.5) kept; accepted row 7 (d=0.3) and noise
    # row 8 dropped; rows outside R_p never come back.
    expected = np.zeros(10, dtype=bool)
    expected[[2, 5]] = True
    np.testing.assert_array_equal(state.residual_mask, expected)


def test_finish_round_rejects_a_verdict_outside_the_categories(output_dir):
    # An unknown label would fall in none of accepted / ambiguous / noise and be dropped
    # like noise, silently; it can only arrive through a hand-edited checkpoint.
    state = init_refinement_state(np.eye(10, 4, dtype=np.float32))
    state.residual_mask[:] = False
    state.residual_mask[[2, 5, 7, 8]] = True
    state.pending, probe_paths = _pending_round(output_dir)
    annotations = [
        (probe_paths[0], "valid"),
        (probe_paths[1], "exo-ambiguous"),
        (probe_paths[2], "noise"),
    ]
    with pytest.raises(ValueError, match=r"\['valid'\] are not in config.annotation_categories"):
        finish_round(state, annotations, PrototypingConfig(), "alice", output_dir)
    assert state.pending is not None and not state.rounds


# --- loop termination ----------------------------------------------------------------------

_SYNTH = SyntheticDatasetConfig(
    n_participants=3, n_action_clusters=3, min_windows=40, max_windows=60, seed=47
)


def _pooled():
    participants = generate_synthetic_embeddings(_SYNTH)
    xs, pids, locs = [], [], []
    for pid, p in participants.items():
        xs.append(p.embeddings)
        pids.append(np.full(len(p.embeddings), pid))
        locs.append(np.arange(len(p.embeddings)))
    return (np.concatenate(xs), np.concatenate(pids), np.concatenate(locs),
            SyntheticFrameSource(participants))


def test_loop_stops_only_when_accepted_exceeds_max_accepted(output_dir):
    X, pids, locs, fs = _pooled()
    config = PrototypingConfig(
        n_clusters_per_round=4, n_rounds=5, faiss_niter=10, faiss_nredo=1,
        n_probe_neighbors=3, n_probe_participants_target=3, max_accepted_prototypes=2,
    )
    state = init_refinement_state(X)
    state.accepted_centroids = np.eye(2, X.shape[1], dtype=X.dtype)  # == K_max: not exceeded
    state.accepted_cluster_types = ["task-definitive"] * 2
    state.accepted_probe_paths = [Path("x")] * 2
    assert len(start_round(state, X, pids, locs, config, fs, output_dir / "r1")) == 4

    state.pending = None
    state.accepted_centroids = np.eye(3, X.shape[1], dtype=X.dtype)  # 3 > 2: exceeded
    assert start_round(state, X, pids, locs, config, fs, output_dir / "r2") == []
    assert state.pending is None


def test_loop_still_runs_when_the_residual_equals_c(output_dir):
    X, pids, locs, fs = _pooled()
    C = 4
    config = PrototypingConfig(
        n_clusters_per_round=C, n_rounds=5, faiss_niter=10, faiss_nredo=1,
        n_probe_neighbors=3, n_probe_participants_target=3,
    )
    state = init_refinement_state(X)
    state.residual_mask[:] = False
    state.residual_mask[:C] = True
    assert len(start_round(state, X, pids, locs, config, fs, output_dir / "r")) == C


# --- consolidation mapping -----------------------------------------------------------------


def test_raw_to_prototype_matches_the_vocabulary_order():
    raw = np.array([[1.0, 0, 0], [0.9, 0.1, 0], [0, 1.0, 0], [0, 0, 1.0], [0.3, 0.3, 0.9]])
    tree = MembershipTree(
        raw_to_pass1=np.array([0, 0, 1, 2, 3]), pass1_to_pass2=np.array([0, 1, 2, 2])
    )
    pass1 = ConsolidationResult(np.array([0, 0, 1, 2, 3]), np.eye(4, 3), {4: 0.4}, 4)
    pass2 = ConsolidationResult(np.array([0, 1, 2, 2]), np.eye(3, 3), {3: 0.3}, 3)
    vocab = assemble_final_vocabulary(
        tree, {0: True, 1: True, 2: False, 3: True}, {0: True, 1: False, 2: False},
        pass1, pass2, raw, ["task-definitive"] * 5,
    )
    assert vocab.source_raw_indices == [[0, 1], [2], [4], [3]]
    # Hand-derived from the entry order above: k0, k1 -> entry 0; k2 -> 1; k4 -> 2; k3 -> 3.
    np.testing.assert_array_equal(vocab.raw_to_prototype, [0, 0, 1, 3, 2])
    np.testing.assert_array_equal(vocab.raw_centroids, raw)


# --- background radius ---------------------------------------------------------------------


def test_background_radius_uses_the_population_std():
    # [0.25, 0.75]: m = 0.5, population sigma = 0.25 -> exactly 0.75 (sample sigma would
    # give 0.5 + 0.3536 = 0.8536).
    thresholds = compute_prototype_thresholds(np.array([0, 0]), np.array([0.25, 0.75]))
    assert thresholds == {0: 0.75}


# --- pipeline wiring of the assignment rule ------------------------------------------------


def test_pipeline_assigns_through_the_accepted_centroids_not_the_prototype_means(
    monkeypatch, output_dir
):
    # On the synthetic fixture the two rules give identical final symbols, so the stored
    # outputs cannot tell them apart; record what the pipeline actually assigns against.
    from tests.integration.test_full_pipeline import (
        _LOOP_CONFIG,
        _SEGMENTATION_CONFIG,
        _SYMBOLIZATION_CONFIG,
        _SYNTH_CONFIG,
        _CyclingAnnotator,
    )
    from recursive_prototyping import pipeline
    from recursive_prototyping.symbolization import assign as assign_module

    calls = []
    real = assign_module.assign_to_prototypes

    def spy(X, centroids, centroid_to_prototype=None):
        calls.append((centroids.shape[0], centroid_to_prototype is not None))
        return real(X, centroids, centroid_to_prototype)

    monkeypatch.setattr(pipeline, "assign_to_prototypes", spy)
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    result = pipeline.run_full_pipeline(
        {pid: p.embeddings for pid, p in participants.items()},
        SyntheticFrameSource(participants), _CyclingAnnotator(),
        _LOOP_CONFIG, _SEGMENTATION_CONFIG, _SYMBOLIZATION_CONFIG,
        output_dir / "run", annotator_id="t",
    )
    K, G = result.accepted_centroids.shape[0], result.vocabulary.shape[0]
    assert K > G, "fixture must consolidate something, else K-row and G-row centroids coincide"
    assert calls, "assign_to_prototypes was never called through pipeline"
    assert all(n_rows == K and has_mapping for n_rows, has_mapping in calls), calls
