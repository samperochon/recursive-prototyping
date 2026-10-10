"""Unit tests for recursive_prototyping.prototyping.refinement."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.annotation_widget import fake_annotate
from recursive_prototyping.prototyping.clustering import assign_to_centroids
from recursive_prototyping.prototyping.refinement import (
    PooledFrameSourceAdapter,
    RefinementState,
    RoundResult,
    build_residual_set,
    finish_round,
    init_refinement_state,
    load_refinement_state,
    run_generative_step,
    run_recursive_loop,
    save_refinement_state,
    split_by_qualification,
    start_round,
)
from recursive_prototyping.synthetic.frames import SyntheticFrameSource
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

# ---------------------------------------------------------------------------
# split_by_qualification
# ---------------------------------------------------------------------------


def _qualification_df() -> pd.DataFrame:
    # alice/round=1: all 5 categories present (one cluster each).
    # alice/round=2: a different, smaller set of clusters -- confirms rounds don't leak.
    # bob/round=1: reuses cluster_id 0 with a DIFFERENT label than alice's -- confirms
    # annotator_id doesn't leak either.
    return pd.DataFrame(
        {
            "cluster_id": [0, 1, 2, 3, 4, 0, 1, 0],
            "round": [1, 1, 1, 1, 1, 2, 2, 1],
            "cluster_type": [
                "task-definitive",
                "exo-definitive",
                "task-ambiguous",
                "exo-ambiguous",
                "noise",
                "noise",
                "task-definitive",
                "task-ambiguous",
            ],
            "annotator_id": ["alice", "alice", "alice", "alice", "alice", "alice", "alice", "bob"],
        }
    )


def test_split_by_qualification_all_five_categories():
    df = _qualification_df()

    accepted, ambiguous, noise = split_by_qualification(df, round_number=1, annotator_id="alice")

    assert accepted == [0, 1]
    assert ambiguous == [2, 3]
    assert noise == [4]


def test_split_by_qualification_different_round_does_not_leak():
    df = _qualification_df()

    accepted, ambiguous, noise = split_by_qualification(df, round_number=2, annotator_id="alice")

    assert accepted == [1]
    assert ambiguous == []
    assert noise == [0]


def test_split_by_qualification_different_annotator_does_not_leak():
    df = _qualification_df()

    # bob only has one row: (cluster_id=0, round=1, "task-ambiguous").
    accepted, ambiguous, noise = split_by_qualification(df, round_number=1, annotator_id="bob")

    assert accepted == []
    assert ambiguous == [0]
    assert noise == []

    # And alice's round=1 result must be unaffected by bob's row sharing cluster_id=0.
    alice_accepted, _, _ = split_by_qualification(df, round_number=1, annotator_id="alice")
    assert alice_accepted == [0, 1]


# ---------------------------------------------------------------------------
# build_residual_set
# ---------------------------------------------------------------------------


def test_build_residual_set_known_mask():
    # cluster 0: accepted -- only its weakly assigned members (distance > d_min) are recycled.
    # cluster 1: ambiguous -- every member is recycled, whatever its distance.
    # cluster 2: noise -- dropped, even the deliberately HIGH distance 0.9 at index 4.
    # cluster 3: ambiguous -- recycled.
    labels = np.array([0, 0, 1, 1, 2, 2, 3])
    distances = np.array([0.50, 0.10, 0.40, 0.20, 0.90, 0.05, 0.35])

    mask = build_residual_set(labels, distances, accepted_ids={0}, ambiguous_ids={1, 3}, d_min=0.3)

    expected = np.array([True, False, True, True, False, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_build_residual_set_with_only_noise_clusters_keeps_nothing():
    labels = np.array([0, 1, 2])
    distances = np.array([0.9, 0.9, 0.9])

    mask = build_residual_set(labels, distances, accepted_ids=set(), ambiguous_ids=set(), d_min=0.1)

    np.testing.assert_array_equal(mask, np.array([False, False, False]))


# ---------------------------------------------------------------------------
# PooledFrameSourceAdapter
# ---------------------------------------------------------------------------


class _RecordingFrameSource:
    """Stub FrameSource that records every (participant_id, embedding_index) it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def central_frame(self, participant_id: str, embedding_index: int) -> np.ndarray:
        self.calls.append((participant_id, embedding_index))
        return np.zeros((2, 2, 3), dtype=np.uint8)


def test_pooled_frame_source_adapter_translates_global_to_local():
    # local_indices[global_row] = that row's local window index within its own participant.
    local_indices = np.array([5, 2, 9, 0])
    base = _RecordingFrameSource()
    adapter = PooledFrameSourceAdapter(local_indices, base)

    adapter.central_frame("p1", 0)
    adapter.central_frame("p2", 3)
    adapter.central_frame("p3", 2)

    assert base.calls == [("p1", 5), ("p2", 0), ("p3", 9)]


# ---------------------------------------------------------------------------
# run_recursive_loop (small synthetic pooled dataset, oracle-driven fake annotator)
# ---------------------------------------------------------------------------

_SYNTH_CONFIG = SyntheticDatasetConfig(
    n_participants=4, n_action_clusters=3, min_windows=60, max_windows=100, seed=47
)


def _pool_participants(participants):
    """Pool all participants' embeddings + oracle labels into parallel global arrays."""
    embeddings_list, participant_ids_list = [], []
    oracle_cluster_ids_list, local_indices_list = [], []

    for participant_id, participant in participants.items():
        n = len(participant.embeddings)
        embeddings_list.append(participant.embeddings)
        participant_ids_list.append(np.full(n, participant_id))
        oracle_cluster_ids_list.append(participant.oracle_cluster_ids)
        local_indices_list.append(np.arange(n))

    return (
        np.concatenate(embeddings_list, axis=0),
        np.concatenate(participant_ids_list, axis=0),
        np.concatenate(oracle_cluster_ids_list, axis=0),
        np.concatenate(local_indices_list, axis=0),
    )


def _loop_config(**overrides) -> PrototypingConfig:
    defaults = dict(
        n_clusters_per_round=8,
        n_rounds=2,
        faiss_niter=20,
        faiss_nredo=1,
        faiss_seed=47,
        n_probe_neighbors=6,
        n_probe_participants_target=4,
        probe_image_ncols=4,
        residual_distance_threshold=0.1,
    )
    defaults.update(overrides)
    return PrototypingConfig(**defaults)


class _OracleDrivenAnnotator:
    """Test-only scripted annotator, built on top of ``fake_annotate`` (never invents a
    filename-parsing scheme -- it relies purely on ``fake_annotate``'s documented
    order-preserving contract, exactly like production code would).

    To label each round's candidate clusters by the DOMINANT oracle cluster id among their
    members, it mirrors run_recursive_loop's own generative step + residual-set update on a
    parallel ``oracle_cluster_ids`` array. This works without run_recursive_loop needing to
    expose any internal state through the ``annotate_fn`` callback (whose real contract is
    strictly `image_paths -> labels`, nothing more) because ``run_generative_step`` and
    ``build_residual_set`` are pure/deterministic given a fixed ``faiss_seed`` -- so this
    shadow computation's residual-set bookkeeping stays exactly in lockstep with
    run_recursive_loop's own internal bookkeeping, round for round.
    """

    def __init__(
        self, X_train: np.ndarray, oracle_cluster_ids: np.ndarray, config: PrototypingConfig
    ) -> None:
        self._X_train = X_train
        self._oracle_cluster_ids = oracle_cluster_ids
        self._config = config
        self._residual_mask = np.ones(len(X_train), dtype=bool)

    def __call__(self, image_paths, categories):
        residual_global_indices = np.flatnonzero(self._residual_mask)
        R_p = self._X_train[residual_global_indices]
        oracle_p = self._oracle_cluster_ids[residual_global_indices]

        centroids, labels, distances = run_generative_step(R_p, self._config)
        assert centroids.shape[0] == len(image_paths)

        cluster_types = []
        for c in range(centroids.shape[0]):
            member_oracle_ids = oracle_p[labels == c]
            if member_oracle_ids.size == 0:
                cluster_types.append("noise")
                continue
            dominant_id, _count = Counter(member_oracle_ids.tolist()).most_common(1)[0]
            cluster_types.append("task-definitive" if dominant_id != -1 else "exo-ambiguous")

        labels_iter = iter(cluster_types)
        result = fake_annotate(image_paths, lambda _path: next(labels_iter))

        accepted_ids = {c for c, label in enumerate(cluster_types) if label == "task-definitive"}
        ambiguous_ids = {c for c, label in enumerate(cluster_types) if label == "exo-ambiguous"}
        residual_mask_local = build_residual_set(
            labels, distances, accepted_ids, ambiguous_ids, self._config.residual_distance_threshold
        )
        next_residual_global = residual_global_indices[residual_mask_local]
        self._residual_mask = np.zeros(len(self._X_train), dtype=bool)
        self._residual_mask[next_residual_global] = True

        return result


def _run_loop(X_train, participant_ids, oracle_cluster_ids, local_indices, config, base_frame_source, run_dir):
    annotate_fn = _OracleDrivenAnnotator(X_train, oracle_cluster_ids, config)
    return run_recursive_loop(
        X_train=X_train,
        participant_ids=participant_ids,
        local_indices=local_indices,
        config=config,
        annotate_fn=annotate_fn,
        base_frame_source=base_frame_source,
        annotator_id="oracle-annotator",
        out_dir=run_dir,
    )


def test_run_recursive_loop_terminates_accumulates_monotonically_and_is_unit_norm(output_dir):
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, participant_ids, oracle_cluster_ids, local_indices = _pool_participants(participants)
    base_frame_source = SyntheticFrameSource(participants)
    config = _loop_config()

    state = _run_loop(
        X_train, participant_ids, oracle_cluster_ids, local_indices, config, base_frame_source,
        output_dir / "run_a",
    )

    # --- terminates after n_rounds=2 ---
    assert [r.round for r in state.rounds] == [1, 2]

    # --- accepted count is monotonically non-decreasing round over round ---
    cumulative = 0
    for round_result in state.rounds:
        accepted_ids, _, _ = split_by_qualification(
            state.qualification_mapping, round_number=round_result.round, annotator_id="oracle-annotator"
        )
        new_cumulative = cumulative + len(accepted_ids)
        assert new_cumulative >= cumulative
        cumulative = new_cumulative

    assert cumulative == len(state.accepted_centroids)
    assert len(state.accepted_centroids) > 0, "sanity: the oracle-driven rule should accept something"
    assert (
        len(state.accepted_centroids)
        == len(state.accepted_cluster_types)
        == len(state.accepted_probe_paths)
    )

    # --- every accepted centroid is unit-norm ---
    norms = np.linalg.norm(state.accepted_centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # --- accepted probe images actually exist (PooledFrameSourceAdapter rendered fine) ---
    for path in state.accepted_probe_paths:
        assert path.exists()

    # --- accepted labels are only the two "-definitive" categories ---
    assert set(state.accepted_cluster_types).issubset({"task-definitive", "exo-definitive"})


def test_run_recursive_loop_deterministic_given_same_seed(output_dir):
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, participant_ids, oracle_cluster_ids, local_indices = _pool_participants(participants)
    base_frame_source = SyntheticFrameSource(participants)
    config = _loop_config()

    state_a = _run_loop(
        X_train, participant_ids, oracle_cluster_ids, local_indices, config, base_frame_source,
        output_dir / "run_a",
    )
    state_b = _run_loop(
        X_train, participant_ids, oracle_cluster_ids, local_indices, config, base_frame_source,
        output_dir / "run_b",
    )

    assert state_a.accepted_centroids.shape == state_b.accepted_centroids.shape
    np.testing.assert_allclose(state_a.accepted_centroids, state_b.accepted_centroids)
    assert state_a.accepted_cluster_types == state_b.accepted_cluster_types


# ---------------------------------------------------------------------------
# save_refinement_state / load_refinement_state
# ---------------------------------------------------------------------------


def test_save_load_refinement_state_roundtrip(output_dir):
    rounds = [
        RoundResult(
            round=1,
            candidate_centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
            cluster_types=["task-definitive", "exo-ambiguous"],
            probe_paths=[Path("round_1_cluster_0.png"), Path("round_1_cluster_1.png")],
        ),
        RoundResult(
            round=2,
            candidate_centroids=np.array([[0.0, -1.0]]),
            cluster_types=["exo-definitive"],
            probe_paths=[Path("round_2_cluster_0.png")],
        ),
    ]
    qualification_mapping = pd.DataFrame(
        {
            "cluster_id": pd.array([0, 1, 0], dtype="int64"),
            "round": pd.array([1, 1, 2], dtype="int64"),
            "cluster_type": ["task-definitive", "exo-ambiguous", "exo-definitive"],
            "annotator_id": ["alice", "alice", "alice"],
        }
    )
    state = RefinementState(
        rounds=rounds,
        accepted_centroids=np.array([[1.0, 0.0], [0.0, -1.0]]),
        accepted_cluster_types=["task-definitive", "exo-definitive"],
        accepted_probe_paths=[Path("round_1_cluster_0.png"), Path("round_2_cluster_0.png")],
        qualification_mapping=qualification_mapping,
    )

    state_dir = output_dir / "refinement_state"
    save_refinement_state(state, state_dir)
    loaded = load_refinement_state(state_dir)

    assert len(loaded.rounds) == len(state.rounds)
    for original, restored in zip(state.rounds, loaded.rounds):
        assert restored.round == original.round
        np.testing.assert_allclose(restored.candidate_centroids, original.candidate_centroids)
        assert restored.cluster_types == original.cluster_types
        assert restored.probe_paths == original.probe_paths

    np.testing.assert_allclose(loaded.accepted_centroids, state.accepted_centroids)
    assert loaded.accepted_cluster_types == state.accepted_cluster_types
    assert loaded.accepted_probe_paths == state.accepted_probe_paths
    pd.testing.assert_frame_equal(loaded.qualification_mapping, state.qualification_mapping)


# ---------------------------------------------------------------------------
# start_round / finish_round -- round-level resumability
# ---------------------------------------------------------------------------


def test_resuming_through_disk_at_every_seam_matches_the_monolithic_loop(output_dir):
    """THE decisive resumability test.

    A live annotation pass is inherently two cells per round, and the kernel can die between
    them, so the claim being tested is not "the loop can be paused" but "the loop can be paused,
    PERSISTED, RELOADED and resumed with no divergence". This drives start_round/finish_round
    with a save+load round trip at BOTH seams -- before each start_round and again in the gap
    between rendering and annotating, which is exactly where a real rater sits -- and requires
    the result to equal run_recursive_loop's.

    It is also what pins ``residual_mask`` and ``pending`` as load-bearing: drop either from
    the persisted state and the reloaded run cannot reproduce the reference.
    """
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, participant_ids, oracle_cluster_ids, local_indices = _pool_participants(participants)
    base_frame_source = SyntheticFrameSource(participants)
    config = _loop_config()

    reference = _run_loop(
        X_train, participant_ids, oracle_cluster_ids, local_indices, config, base_frame_source,
        output_dir / "reference",
    )

    annotate_fn = _OracleDrivenAnnotator(X_train, oracle_cluster_ids, config)
    run_dir = output_dir / "resumed"
    state_dir = output_dir / "resumed_state"

    state = init_refinement_state(X_train)
    n_rounds_driven = 0
    while True:
        # --- kernel restart #1: between rounds ---
        save_refinement_state(state, state_dir)
        state = load_refinement_state(state_dir)

        probe_paths = start_round(
            state, X_train, participant_ids, local_indices, config, base_frame_source, run_dir
        )
        if not probe_paths:
            break

        # --- kernel restart #2: rendered, awaiting a human verdict ---
        assert state.pending is not None
        save_refinement_state(state, state_dir)
        assert (state_dir / "pending.npz").exists()
        state = load_refinement_state(state_dir)
        assert state.pending is not None
        assert state.pending.probe_paths == probe_paths

        annotations = annotate_fn(probe_paths, config.annotation_categories)
        finish_round(state, annotations, config, "oracle-annotator", run_dir)
        assert state.pending is None
        save_refinement_state(state, state_dir)
        assert not (state_dir / "pending.npz").exists(), (
            "a finished round must clear pending.npz -- a stale one reads as 'awaiting "
            "verdicts' to anyone inspecting the state directory"
        )
        n_rounds_driven += 1

    assert n_rounds_driven == len(reference.rounds)
    np.testing.assert_allclose(state.accepted_centroids, reference.accepted_centroids)
    assert state.accepted_cluster_types == reference.accepted_cluster_types
    assert [r.round for r in state.rounds] == [r.round for r in reference.rounds]
    for resumed_round, reference_round in zip(state.rounds, reference.rounds):
        assert resumed_round.cluster_types == reference_round.cluster_types
        np.testing.assert_allclose(
            resumed_round.candidate_centroids, reference_round.candidate_centroids
        )
    pd.testing.assert_frame_equal(
        state.qualification_mapping.reset_index(drop=True),
        reference.qualification_mapping.reset_index(drop=True),
    )


def test_start_round_refuses_while_a_round_is_pending(output_dir):
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, participant_ids, oracle_cluster_ids, local_indices = _pool_participants(participants)
    base_frame_source = SyntheticFrameSource(participants)
    config = _loop_config()

    state = init_refinement_state(X_train)
    args = (X_train, participant_ids, local_indices, config, base_frame_source, output_dir / "run")
    start_round(state, *args)

    with pytest.raises(ValueError, match="already rendered"):
        start_round(state, *args)


def test_finish_round_refuses_when_nothing_is_pending(output_dir):
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, _participant_ids, _oracle, _local_indices = _pool_participants(participants)
    config = _loop_config()

    state = init_refinement_state(X_train)
    with pytest.raises(ValueError, match="no round is pending"):
        finish_round(state, [], config, "oracle-annotator", output_dir / "run")


def test_start_round_refuses_a_state_that_carries_no_residual_mask(output_dir):
    """A state directory without residual_mask.npy loads with residual_mask=None. It
    is still inspectable and consolidatable, but resuming it would silently restart from R_1,
    so start_round must refuse rather than quietly do the wrong thing.
    """
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, participant_ids, _oracle, local_indices = _pool_participants(participants)
    base_frame_source = SyntheticFrameSource(participants)
    config = _loop_config()

    state = RefinementState(
        rounds=[],
        accepted_centroids=np.empty((0, X_train.shape[1])),
        accepted_cluster_types=[],
        accepted_probe_paths=[],
        qualification_mapping=pd.DataFrame(
            columns=["cluster_id", "round", "cluster_type", "annotator_id"]
        ),
    )
    assert state.residual_mask is None

    with pytest.raises(ValueError, match="residual_mask is None"):
        start_round(
            state, X_train, participant_ids, local_indices, config, base_frame_source,
            output_dir / "run",
        )


def test_a_state_dir_without_the_resumability_files_still_loads(output_dir):
    """save/load must not break on a state directory that holds neither residual_mask.npy
    nor pending.npz.
    """
    state = RefinementState(
        rounds=[],
        accepted_centroids=np.array([[1.0, 0.0]]),
        accepted_cluster_types=["task-definitive"],
        accepted_probe_paths=[Path("round_1_cluster_0.png")],
        qualification_mapping=pd.DataFrame(
            columns=["cluster_id", "round", "cluster_type", "annotator_id"]
        ),
    )
    state_dir = output_dir / "unversioned_state"
    save_refinement_state(state, state_dir)
    # Neither resumability file is present.
    assert not (state_dir / "pending.npz").exists()
    (state_dir / "residual_mask.npy").unlink(missing_ok=True)

    loaded = load_refinement_state(state_dir)
    assert loaded.residual_mask is None
    assert loaded.pending is None
    assert loaded.next_round == 1
    np.testing.assert_allclose(loaded.accepted_centroids, state.accepted_centroids)


def test_start_round_terminates_cleanly_when_the_residual_falls_below_c(output_dir):
    """The loop must END when the residual is too small for a C-cluster k-means, not crash.

    faiss raises an opaque RuntimeError for ``0 < nx < k``, and the residual shrinks every
    round, so that band is where a multi-round run can land -- after the annotator's time
    has been spent. The guard is
    therefore checked here against the real faiss backend rather than a mock: a mocked
    generative step would pass even if the floor were removed.
    """
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, participant_ids, _oracle, local_indices = _pool_participants(participants)
    base_frame_source = SyntheticFrameSource(participants)

    # C strictly greater than the residual, but the residual is NOT empty -- the crashing band.
    n_survivors = 5
    config = _loop_config()
    config = PrototypingConfig(
        **{**config.__dict__, "n_clusters_per_round": n_survivors + 1}
    )

    state = init_refinement_state(X_train)
    state.residual_mask = np.zeros(X_train.shape[0], dtype=bool)
    state.residual_mask[:n_survivors] = True

    assert start_round(
        state, X_train, participant_ids, local_indices, config, base_frame_source,
        output_dir / "run",
    ) == []
    assert state.pending is None, "a terminated round must not be left pending"
    assert state.next_round == 1, "termination must not consume a round"


def test_faiss_really_does_raise_below_c_so_the_floor_is_load_bearing():
    """Pins the reason the floor above exists. If a future faiss silently tolerates nx < k this
    test fails, and the guard's justification -- not the guard -- is what needs revisiting.
    """
    config = _loop_config()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((9, 16)).astype(np.float32)

    with pytest.raises(RuntimeError):
        run_generative_step(
            X, PrototypingConfig(**{**config.__dict__, "n_clusters_per_round": 10})
        )


def test_run_generative_step_numbers_clusters_by_decreasing_size():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 16)).astype(np.float32)

    centroids, labels, distances = run_generative_step(X, _loop_config())

    sizes = np.bincount(labels, minlength=centroids.shape[0])
    assert (np.diff(sizes) <= 0).all()
    # Relabeling kept each row attached to its own centroid.
    relabeled, redistances = assign_to_centroids(X, centroids)
    np.testing.assert_array_equal(relabeled, labels)
    np.testing.assert_allclose(redistances, distances, rtol=1e-5)
