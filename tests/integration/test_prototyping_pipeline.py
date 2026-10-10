"""Integration test for prototyping: recursive centroid
accumulation (``refinement.py``) followed by two-pass HAC vocabulary
consolidation (``consolidation.py``), exercised end-to-end on a small
synthetic pooled multi-participant dataset.

Covers, in order (docs/METHOD.md, section 1):
generate_synthetic_embeddings -> pool participants into (X_train, participant_ids,
local_indices) -> run_recursive_loop (fake_annotate driven by an oracle-derived 5-way
rule) -> RefinementState -> run_two_stage_consolidation (per-group validity computed
directly from the HAC/silhouette result -- no second annotation round) ->
assemble_final_vocabulary -> Vocabulary, plus a determinism check on the whole
(refinement + consolidation) pipeline and an explicit check that
``PooledFrameSourceAdapter`` is doing real, load-bearing global-row ->
participant-local-index translation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.annotation_widget import fake_annotate
from recursive_prototyping.prototyping.consolidation import (
    ConsolidationResult,
    Vocabulary,
    assemble_final_vocabulary,
    consolidate_once,
    run_two_stage_consolidation,
)
from recursive_prototyping.prototyping.refinement import (
    PooledFrameSourceAdapter,
    RefinementState,
    build_residual_set,
    run_generative_step,
    run_recursive_loop,
)
from recursive_prototyping.synthetic.frames import SyntheticFrameSource
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

_SYNTH_CONFIG = SyntheticDatasetConfig(
    n_participants=5, n_action_clusters=4, min_windows=50, max_windows=90, seed=47
)

_LOOP_CONFIG = PrototypingConfig(
    n_clusters_per_round=10,
    n_rounds=3,
    faiss_niter=25,
    faiss_nredo=1,
    faiss_seed=47,
    n_probe_neighbors=6,
    n_probe_participants_target=4,
    probe_image_ncols=5,
    residual_distance_threshold=0.1,
)

# 5-way oracle-driven annotator's purity rule.
_PURITY_THRESHOLD = 0.6
_MIN_CLUSTER_SIZE_FOR_LABEL = 3


def _pool_participants(participants):
    """Pool all participants' embeddings + oracle labels into parallel global arrays.

    local_indices[global_row] = that row's position within its own participant's
    original sequence -- the pooled-indexing bookkeeping that every caller of the
    recursive loop has to build alongside participant_ids.
    """
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


class _OracleDrivenFiveWayAnnotator:
    """Deterministic 5-way scripted annotator for ``run_recursive_loop``, built on top of
    ``fake_annotate`` (never invents a filename-parsing scheme -- it relies purely
    on ``fake_annotate``'s documented order-preserving contract).

    To label each round's candidate clusters, it mirrors run_recursive_loop's own generative
    step + residual-set update on a parallel ``oracle_cluster_ids`` array -- the same
    shadow-computation pattern as ``tests/unit/test_refinement.py``'s ``_OracleDrivenAnnotator``,
    extended here from a 2-way task/exo-ambiguous stand-in to the FULL 5-way
    task/exo-definitive/ambiguous/noise scheme, using a purity heuristic:

      - fewer than _MIN_CLUSTER_SIZE_FOR_LABEL members  -> "noise" (too small to judge).
      - else purity = (count of the cluster's dominant oracle_cluster_id) / (cluster size):
          dominant_id == -1 (background) and purity >= threshold -> "exo-definitive"
          dominant_id == -1 (background) and purity <  threshold -> "exo-ambiguous"
          dominant_id != -1 (real action) and purity >= threshold -> "task-definitive"
          dominant_id != -1 (real action) and purity <  threshold -> "task-ambiguous"

    This works without run_recursive_loop needing to expose any internal state through the
    annotate_fn callback (whose real contract is strictly `image_paths -> labels`) because
    run_generative_step/build_residual_set are pure/deterministic given a fixed faiss_seed --
    so this shadow computation's residual-set bookkeeping stays exactly in lockstep with
    run_recursive_loop's own internal bookkeeping, round for round.
    """

    def __init__(
        self, X_train: np.ndarray, oracle_cluster_ids: np.ndarray, config: PrototypingConfig
    ) -> None:
        self._X_train = X_train
        self._oracle_cluster_ids = oracle_cluster_ids
        self._config = config
        self._residual_mask = np.ones(len(X_train), dtype=bool)

    @staticmethod
    def _label_for_cluster(member_oracle_ids: np.ndarray) -> str:
        if member_oracle_ids.size < _MIN_CLUSTER_SIZE_FOR_LABEL:
            return "noise"
        dominant_id, dominant_count = Counter(member_oracle_ids.tolist()).most_common(1)[0]
        purity = dominant_count / member_oracle_ids.size
        if dominant_id == -1:
            return "exo-definitive" if purity >= _PURITY_THRESHOLD else "exo-ambiguous"
        return "task-definitive" if purity >= _PURITY_THRESHOLD else "task-ambiguous"

    def __call__(self, image_paths, categories):
        residual_global_indices = np.flatnonzero(self._residual_mask)
        R_p = self._X_train[residual_global_indices]
        oracle_p = self._oracle_cluster_ids[residual_global_indices]

        centroids, labels, distances = run_generative_step(R_p, self._config)
        assert centroids.shape[0] == len(image_paths)

        cluster_types = [
            self._label_for_cluster(oracle_p[labels == c]) for c in range(centroids.shape[0])
        ]

        labels_iter = iter(cluster_types)
        result = fake_annotate(image_paths, lambda _path: next(labels_iter))

        ambiguous_ids = {
            c
            for c, label in enumerate(cluster_types)
            if label in ("task-ambiguous", "exo-ambiguous")
        }
        accepted_ids = {
            c
            for c, label in enumerate(cluster_types)
            if label in ("task-definitive", "exo-definitive")
        }
        residual_mask_local = build_residual_set(
            labels, distances, accepted_ids, ambiguous_ids, self._config.residual_distance_threshold
        )
        next_residual_global = residual_global_indices[residual_mask_local]
        self._residual_mask = np.zeros(len(self._X_train), dtype=bool)
        self._residual_mask[next_residual_global] = True

        return result


def _shadow_consolidation_passes(accepted_centroids: np.ndarray, config: PrototypingConfig):
    """Independently replay the same two HAC passes ``run_two_stage_consolidation`` runs
    internally, purely as a cross-check that the real function's assignments match.
    ``consolidate_once`` is pure/deterministic given fixed centroids, so this shadow
    computation must agree exactly with what ``run_two_stage_consolidation`` produces.
    """
    pass1 = consolidate_once(accepted_centroids, config.consolidation_linkage)
    pass2 = consolidate_once(pass1.group_centroids, config.consolidation_linkage)
    return pass1, pass2


def _run_full_pipeline(out_dir: Path):
    """Full prototyping chain: synthetic data -> pooling -> recursive loop -> two-stage
    consolidation -> final vocabulary. Returns everything a test might want to inspect.
    """
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    X_train, participant_ids, oracle_cluster_ids, local_indices = _pool_participants(participants)
    base_frame_source = SyntheticFrameSource(participants)

    refinement_annotate_fn = _OracleDrivenFiveWayAnnotator(X_train, oracle_cluster_ids, _LOOP_CONFIG)
    state = run_recursive_loop(
        X_train=X_train,
        participant_ids=participant_ids,
        local_indices=local_indices,
        config=_LOOP_CONFIG,
        annotate_fn=refinement_annotate_fn,
        base_frame_source=base_frame_source,
        annotator_id="oracle-annotator",
        out_dir=out_dir / "refinement",
    )

    shadow_pass1, shadow_pass2 = _shadow_consolidation_passes(state.accepted_centroids, _LOOP_CONFIG)
    pass1, pass2, tree, pass1_validity, pass2_validity = run_two_stage_consolidation(
        state.accepted_centroids, _LOOP_CONFIG
    )

    # Sanity: the shadow HAC replay must match the actual passes run_two_stage_consolidation
    # performed internally (both are pure functions of the same accepted_centroids/config, so
    # they must agree exactly).
    np.testing.assert_array_equal(shadow_pass1.assignment, pass1.assignment)
    np.testing.assert_array_equal(shadow_pass2.assignment, pass2.assignment)

    vocabulary = assemble_final_vocabulary(
        tree,
        pass1_validity,
        pass2_validity,
        pass1,
        pass2,
        state.accepted_centroids,
        state.accepted_cluster_types,
    )

    return {
        "participants": participants,
        "X_train": X_train,
        "participant_ids": participant_ids,
        "local_indices": local_indices,
        "base_frame_source": base_frame_source,
        "state": state,
        "vocabulary": vocabulary,
    }


def test_prototyping_pipeline_end_to_end(output_dir):
    result = _run_full_pipeline(output_dir / "run_a")
    state: RefinementState = result["state"]
    vocabulary: Vocabulary = result["vocabulary"]

    # --- sanity: the oracle-driven rule accepted something across the 3 rounds ---
    n_accepted = state.accepted_centroids.shape[0]
    assert n_accepted > 0, "sanity: the oracle-driven rule should accept something"
    assert len(state.accepted_cluster_types) == n_accepted
    assert len(state.accepted_probe_paths) == n_accepted
    assert set(state.accepted_cluster_types).issubset({"task-definitive", "exo-definitive"})

    # --- final vocabulary size G <= number of accepted raw centroids ---
    G = vocabulary.centroids.shape[0]
    assert G <= n_accepted
    assert G > 0
    assert len(vocabulary.cluster_types) == G
    assert len(vocabulary.source_raw_indices) == G

    # --- every final vocabulary centroid is unit-norm ---
    norms = np.linalg.norm(vocabulary.centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # --- source_raw_indices partitions all raw accepted centroid indices: covers every
    # raw index exactly once, no index appearing under two different vocabulary entries ---
    all_source_indices = [idx for group in vocabulary.source_raw_indices for idx in group]
    assert sorted(all_source_indices) == list(range(n_accepted))
    assert len(all_source_indices) == len(set(all_source_indices))

    # --- PooledFrameSourceAdapter: a real (participant_id, global_index) ->
    # (participant_id, local_index) translation, exercised end-to-end ---
    # (a) run_recursive_loop itself only succeeded (every accepted probe image rendered
    #     without an IndexError) because it internally built and used a
    #     PooledFrameSourceAdapter to translate pooled global row positions back to each
    #     participant's own local window index before calling SyntheticFrameSource.
    for path in state.accepted_probe_paths:
        assert path.exists()

    # (b) explicit, direct proof that the translation is load-bearing (not a no-op): take
    # the LAST row of the pooled X_train (a global row position that is only valid for the
    # last participant's OWN local window index once translated) and show that calling the
    # base FrameSource directly with the untranslated global index is out of range for that
    # participant's own embedding sequence, whereas going through PooledFrameSourceAdapter
    # (the real class from refinement.py, not a test double) with the SAME pooled
    # local_indices array the recursive loop itself builds succeeds and returns a valid frame.
    X_train = result["X_train"]
    participant_ids = result["participant_ids"]
    local_indices = result["local_indices"]
    base_frame_source = result["base_frame_source"]
    participants = result["participants"]

    last_global_row = len(X_train) - 1
    last_participant_id = str(participant_ids[last_global_row])
    last_local_index = int(local_indices[last_global_row])
    n_windows_last_participant = len(participants[last_participant_id].embeddings)

    assert last_local_index < n_windows_last_participant
    assert last_global_row >= n_windows_last_participant, (
        "test precondition: the pooled dataset must be large enough that the global row "
        "position genuinely differs from (and exceeds) the last participant's own sequence "
        "length, otherwise the untranslated call below would not actually be out of range"
    )

    try:
        base_frame_source.central_frame(last_participant_id, last_global_row)
        untranslated_raised = False
    except IndexError:
        untranslated_raised = True
    assert untranslated_raised, (
        "expected calling the base FrameSource with an UNtranslated global row index to be "
        "out of range for the last participant's own (shorter) embedding sequence -- if this "
        "didn't raise, the test does not prove the translation is necessary"
    )

    adapter = PooledFrameSourceAdapter(local_indices, base_frame_source)
    translated_frame = adapter.central_frame(last_participant_id, last_global_row)
    directly_translated_frame = base_frame_source.central_frame(
        last_participant_id, last_local_index
    )
    np.testing.assert_array_equal(translated_frame, directly_translated_frame)


def test_prototyping_pipeline_deterministic(output_dir):
    result_a = _run_full_pipeline(output_dir / "run_a")
    result_b = _run_full_pipeline(output_dir / "run_b")

    vocab_a: Vocabulary = result_a["vocabulary"]
    vocab_b: Vocabulary = result_b["vocabulary"]

    assert vocab_a.centroids.shape == vocab_b.centroids.shape
    np.testing.assert_allclose(vocab_a.centroids, vocab_b.centroids)
    assert vocab_a.cluster_types == vocab_b.cluster_types
    assert vocab_a.source_raw_indices == vocab_b.source_raw_indices

    state_a: RefinementState = result_a["state"]
    state_b: RefinementState = result_b["state"]
    np.testing.assert_allclose(state_a.accepted_centroids, state_b.accepted_centroids)
    assert state_a.accepted_cluster_types == state_b.accepted_cluster_types
