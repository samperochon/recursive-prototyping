"""Integration test for the full pipeline: ``run_full_pipeline`` wiring prototyping
(pooled recursive loop + two-stage consolidation into the vocabulary), segmentation
(``calibrate_and_segment`` on the same per-participant embeddings), and symbolization
(assign -> background_filter -> symbolize, per participant) end-to-end on the repo's synthetic dataset, plus:

- a check that ``pipeline.symbolize_participant`` threads
  ``SymbolizationConfig.background_n_sigma`` into ``background_filter`` (a
  hardcoded ``n_sigma=1.0`` in pipeline.py fails ``test_background_n_sigma_is_threaded``),
- artifact persistence (npy/npz/parquet/json only, everything reloadable via
  the io helpers, no pickle anywhere under out_dir), and
- a determinism check (two runs into different out_dirs -> identical
  vocabulary/cpts/symbols).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from recursive_prototyping.config.schema import (
    PrototypingConfig,
    SegmentationConfig,
    SymbolizationConfig,
)
from recursive_prototyping.io.arrays import load_array, load_arrays
from recursive_prototyping.pipeline import (
    PipelineResult,
    pooled_prototype_thresholds,
    run_full_pipeline,
    symbolize_participant,
)
from recursive_prototyping.prototyping.annotation_widget import fake_annotate
from recursive_prototyping.prototyping.refinement import load_refinement_state
from recursive_prototyping.synthetic.frames import SyntheticFrameSource
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

_SYNTH_CONFIG = SyntheticDatasetConfig(
    n_participants=3,
    n_action_clusters=3,
    min_windows=60,
    max_windows=100,
    embedding_dim=32,
    seed=47,
)

# Fast settings mirroring test_prototyping_pipeline.py's.
_LOOP_CONFIG = PrototypingConfig(
    n_clusters_per_round=12,
    n_rounds=2,
    faiss_niter=50,
    faiss_nredo=1,
    faiss_seed=47,
    n_probe_neighbors=6,
    n_probe_participants_target=3,
    probe_image_ncols=5,
)

# Stock defaults, including gamma_temperature=1.0.
_SEGMENTATION_CONFIG = SegmentationConfig()

# Defaults: background_n_sigma=1.0, min_segment_length=4.
_SYMBOLIZATION_CONFIG = SymbolizationConfig()

_ANNOTATOR_ID = "pipeline-tester"

# Fields persisted per participant in symbols.npz, keyed "{participant_id}__{field}".
_NPZ_FIELDS = ("segmentation_cpts", "cpts", "symbols", "window_symbols")


class _CyclingAnnotator:
    """Deterministic annotate_fn built on fake_annotate's documented order-preserving
    contract (never parses filenames): labels each round's candidate clusters in
    order, cycling task-definitive / exo-definitive / task-ambiguous. So 2/3 of every
    round's clusters are accepted (both accepted types reach the vocabulary) while the
    remaining third re-feeds the residual set, keeping round 2 non-trivial.
    """

    _CYCLE = ("task-definitive", "exo-definitive", "task-ambiguous")

    def __init__(self) -> None:
        self._n_labeled = 0

    def _rule(self, _path) -> str:
        label = self._CYCLE[self._n_labeled % len(self._CYCLE)]
        self._n_labeled += 1
        return label

    def __call__(self, image_paths, categories):
        return fake_annotate(image_paths, self._rule)


def _run_pipeline(out_dir):
    """One full run on the synthetic dataset; returns everything the tests inspect."""
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    embeddings = {pid: p.embeddings for pid, p in participants.items()}
    result = run_full_pipeline(
        embeddings,
        SyntheticFrameSource(participants),
        _CyclingAnnotator(),
        _LOOP_CONFIG,
        _SEGMENTATION_CONFIG,
        _SYMBOLIZATION_CONFIG,
        out_dir,
        annotator_id=_ANNOTATOR_ID,
    )
    return embeddings, result, out_dir


@pytest.fixture(scope="module")
def run_a(tmp_path_factory):
    return _run_pipeline(tmp_path_factory.mktemp("full_pipeline_run_a"))


@pytest.fixture(scope="module")
def run_b(tmp_path_factory):
    return _run_pipeline(tmp_path_factory.mktemp("full_pipeline_run_b"))


def _assert_repo_cpts_convention(cpts: np.ndarray, n_windows: int) -> None:
    """Repo change-point convention: [0, t1, ..., T] int64, strictly increasing."""
    assert cpts.dtype == np.int64
    assert cpts[0] == 0
    assert cpts[-1] == n_windows
    assert np.all(np.diff(cpts) > 0)


def test_full_pipeline_end_to_end(run_a):
    embeddings, result, out_dir = run_a
    assert isinstance(result, PipelineResult)

    # --- vocabulary: non-empty 2-D float, unit-norm rows, typed entries ---
    vocabulary = result.vocabulary
    assert vocabulary.ndim == 2
    n_prototypes = vocabulary.shape[0]
    assert n_prototypes > 0
    assert np.issubdtype(vocabulary.dtype, np.floating)
    np.testing.assert_allclose(np.linalg.norm(vocabulary, axis=1), 1.0, atol=1e-5)
    assert len(result.vocabulary_cluster_types) == n_prototypes
    assert set(result.vocabulary_cluster_types).issubset(
        {"task-definitive", "exo-definitive"}
    )

    # --- per participant: cpts convention, symbol pairing, symbol range ---
    assert set(result.participants) == set(embeddings)
    for participant_id, res in result.participants.items():
        n_windows = len(embeddings[participant_id])
        _assert_repo_cpts_convention(res.segmentation_cpts, n_windows)
        _assert_repo_cpts_convention(res.cpts, n_windows)
        # Non-degenerate segmentation at stock defaults: every participant gets
        # at least one interior change point.
        assert len(res.segmentation_cpts) > 2

        assert res.symbols.dtype == np.int64
        assert len(res.symbols) == len(res.cpts) - 1
        assert np.all((res.symbols >= -1) & (res.symbols < n_prototypes))

        np.testing.assert_array_equal(
            res.window_symbols, np.repeat(res.symbols, np.diff(res.cpts))
        )

        assert res.penalty > 0
        assert res.gamma is not None and res.gamma > 0  # rbf kernel

    # --- artifacts: everything declared exists and reloads via the io helpers ---
    for path in result.artifact_paths.values():
        assert path.exists()

    reloaded_vocabulary = load_array(result.artifact_paths["vocabulary"])
    np.testing.assert_allclose(reloaded_vocabulary, vocabulary)

    # --- the reduction mapping: every accepted centroid maps to one vocabulary row ---
    mapping = load_array(result.artifact_paths["centroid_to_prototype"])
    accepted = load_array(result.artifact_paths["accepted_centroids"])
    assert mapping.shape == (accepted.shape[0],)
    assert set(mapping.tolist()) == set(range(n_prototypes))

    reloaded_arrays = load_arrays(result.artifact_paths["symbols"])
    expected_keys = {
        f"{pid}__{field}" for pid in embeddings for field in _NPZ_FIELDS
    }
    assert set(reloaded_arrays) == expected_keys
    for participant_id, res in result.participants.items():
        for field in _NPZ_FIELDS:
            np.testing.assert_array_equal(
                reloaded_arrays[f"{participant_id}__{field}"], getattr(res, field)
            )

    with result.artifact_paths["calibration_summary"].open() as f:
        summary = json.load(f)
    assert set(summary) == {"slope", "r_squared", "participants"}
    for participant_id, res in result.participants.items():
        entry = summary["participants"][participant_id]
        assert entry["penalty"] == pytest.approx(res.penalty)
        assert entry["gamma"] == pytest.approx(res.gamma)
        assert entry["n_windows"] == len(embeddings[participant_id])

    for name, config in (
        ("prototyping_config", _LOOP_CONFIG),
        ("segmentation_config", _SEGMENTATION_CONFIG),
        ("symbolization_config", _SYMBOLIZATION_CONFIG),
    ):
        with result.artifact_paths[name].open() as f:
            snapshot = json.load(f)
        assert snapshot["random_seed"] == config.random_seed
        if name == "segmentation_config":
            assert snapshot["gamma_temperature"] == 1.0

    state = load_refinement_state(result.artifact_paths["refinement_state"])
    assert state.accepted_centroids.shape[0] >= n_prototypes
    assert state.accepted_centroids.shape[1] == _SYNTH_CONFIG.embedding_dim

    # --- no pickle anywhere under out_dir ---
    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.pickle"))


def test_background_n_sigma_is_threaded(run_a):
    """pipeline.py must pass config.background_n_sigma into the threshold computation. n_sigma=1e6 puts every
    per-prototype radius far above any cosine distance (max 2), so it must produce
    ZERO -1 relabels, while the default 1.0 produces SOME on this data -- a hardcoded
    n_sigma=1.0 inside pipeline.py would fail the huge-value half of this test.
    """
    embeddings, result, _ = run_a

    default_config = SymbolizationConfig(background_n_sigma=1.0)
    huge_config = SymbolizationConfig(background_n_sigma=1e6)
    mapping = result.centroid_to_prototype
    default_thresholds = pooled_prototype_thresholds(
        embeddings, result.accepted_centroids, default_config, mapping
    )
    huge_thresholds = pooled_prototype_thresholds(
        embeddings, result.accepted_centroids, huge_config, mapping
    )

    total_default_relabels = 0
    total_huge_relabels = 0
    for participant_id, X in embeddings.items():
        res = result.participants[participant_id]
        filtered_default, cpts_default, symbols_default = symbolize_participant(
            X, result.accepted_centroids, res.segmentation_cpts,
            default_config, default_thresholds, mapping,
        )
        filtered_huge, _, _ = symbolize_participant(
            X, result.accepted_centroids, res.segmentation_cpts,
            huge_config, huge_thresholds, mapping,
        )
        total_default_relabels += int((filtered_default == -1).sum())
        total_huge_relabels += int((filtered_huge == -1).sum())

        # run_full_pipeline itself was run with the default config, so its stored
        # outputs must match the default-n_sigma leg exactly (same threaded path).
        np.testing.assert_array_equal(cpts_default, res.cpts)
        np.testing.assert_array_equal(symbols_default, res.symbols)

    assert total_huge_relabels == 0
    assert total_default_relabels > 0

    # And the default-config pipeline outputs reflect the filtering downstream: some
    # background symbols survive to the final strings. About half of this data is
    # background and it sits close to the thresholds, so partially-kept background is
    # expected, not asserted against.
    assert any((res.symbols == -1).any() for res in result.participants.values())


def test_background_thresholds_are_cohort_pooled(run_a):
    """One radius per prototype, computed over the WHOLE cohort (docs/METHOD.md,
    section 3).

    Computing m_g + sigma_g from each participant's own windows would give the
    same prototype a different radius per participant. This test proves the
    pooling is load-bearing: the pooled radii must differ from the
    per-participant ones, and the pipeline's stored output must follow the
    POOLED leg, not the per-participant one.
    """
    embeddings, result, _ = run_a
    config = SymbolizationConfig()

    mapping = result.centroid_to_prototype
    pooled = pooled_prototype_thresholds(embeddings, result.accepted_centroids, config, mapping)

    per_participant_differs = False
    for participant_id, X in embeddings.items():
        res = result.participants[participant_id]
        # Per-participant radii are what symbolize_participant computes with
        # thresholds=None.
        _, _, symbols_local = symbolize_participant(
            X, result.accepted_centroids, res.segmentation_cpts, config, None, mapping
        )
        _, _, symbols_pooled = symbolize_participant(
            X, result.accepted_centroids, res.segmentation_cpts, config, pooled, mapping
        )
        if not np.array_equal(symbols_local, symbols_pooled):
            per_participant_differs = True
        # EVERY participant's stored output must follow the pooled leg -- an
        # `any` here would still pass if the pipeline used
        # per-participant scope and merely happened to agree on one participant.
        np.testing.assert_array_equal(symbols_pooled, res.symbols)

    assert per_participant_differs, (
        "pooled and per-participant radii produced identical symbols on every "
        "participant -- this test cannot tell cohort scope from "
        "per-participant scope"
    )

    # The pooled radii are a flat prototype -> float mapping with no participant
    # dimension.
    assert set(pooled) <= set(range(result.vocabulary.shape[0]))
    assert all(isinstance(v, float) for v in pooled.values())


def test_full_pipeline_deterministic(run_a, run_b):
    _, result_a, _ = run_a
    _, result_b, _ = run_b

    assert result_a.vocabulary.shape == result_b.vocabulary.shape
    np.testing.assert_allclose(result_a.vocabulary, result_b.vocabulary)
    assert result_a.vocabulary_cluster_types == result_b.vocabulary_cluster_types

    assert set(result_a.participants) == set(result_b.participants)
    for participant_id, res_a in result_a.participants.items():
        res_b = result_b.participants[participant_id]
        np.testing.assert_array_equal(res_a.segmentation_cpts, res_b.segmentation_cpts)
        np.testing.assert_array_equal(res_a.cpts, res_b.cpts)
        np.testing.assert_array_equal(res_a.symbols, res_b.symbols)
        np.testing.assert_array_equal(res_a.window_symbols, res_b.window_symbols)
        assert res_a.penalty == pytest.approx(res_b.penalty)
        assert res_a.gamma == pytest.approx(res_b.gamma)
