"""Integration test for clustering and probe rendering: cosine k-means
+ Visual Probing Mechanism, exercised end-to-end on synthetic data.

Covers, in order (docs/METHOD.md, section 1):
fit_cosine_kmeans -> assign_to_centroids -> compute_cluster_stats -> render_probe_image
(nearest + furthest probes via sample_probe_indices, rendered with SyntheticFrameSource)
-> fake_annotate -> save_qualification_mapping/load_qualification_mapping round-trip,
plus a determinism check on the whole pipeline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.annotation_io import (
    QualificationRecord,
    load_qualification_mapping,
    save_qualification_mapping,
)
from recursive_prototyping.prototyping.annotation_widget import fake_annotate
from recursive_prototyping.prototyping.clustering import assign_to_centroids, fit_cosine_kmeans
from recursive_prototyping.prototyping.vpm import (
    compute_cluster_stats,
    render_probe_image,
    sample_probe_indices,
)
from recursive_prototyping.synthetic.frames import SyntheticFrameSource
from recursive_prototyping.synthetic.generator import (
    SyntheticDatasetConfig,
    generate_synthetic_embeddings,
)

_SYNTH_CONFIG = SyntheticDatasetConfig(
    n_participants=4, n_action_clusters=3, min_windows=50, max_windows=100
)


class _PooledFrameSourceAdapter:
    """Adapts a participant-local-index FrameSource (``SyntheticFrameSource``) so it can be
    driven by *pooled* array row positions.

    ``sample_probe_indices``/``ProbeSample.embedding_index`` is documented (vpm.py) as "the
    row's position within the arrays passed in" -- when those arrays are a cross-participant
    pool, that position is a global pooled row index, not the participant-local window index
    ``SyntheticFrameSource`` expects. ``local_indices[pooled_position]`` recovers the
    participant-local index for that pooled row (per vpm.py's documented caller
    responsibility to make this translation).
    """

    def __init__(self, base_source: SyntheticFrameSource, local_indices: np.ndarray):
        self._base_source = base_source
        self._local_indices = local_indices

    def central_frame(self, participant_id: str, embedding_index: int) -> np.ndarray:
        local_index = int(self._local_indices[embedding_index])
        return self._base_source.central_frame(participant_id, local_index)


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


def _run_pipeline(config: PrototypingConfig) -> dict:
    """Generate synthetic data (deterministic for a fixed SyntheticDatasetConfig.seed) and
    run fit_cosine_kmeans + assign_to_centroids on the pooled embeddings."""
    participants = generate_synthetic_embeddings(_SYNTH_CONFIG)
    embeddings, participant_ids, oracle_cluster_ids, local_indices = _pool_participants(
        participants
    )

    centroids, _ = fit_cosine_kmeans(
        embeddings, n_clusters=config.n_clusters_per_round, config=config
    )
    labels, distances = assign_to_centroids(embeddings, centroids)

    return {
        "participants": participants,
        "participant_ids": participant_ids,
        "oracle_cluster_ids": oracle_cluster_ids,
        "local_indices": local_indices,
        "centroids": centroids,
        "labels": labels,
        "distances": distances,
    }


def _dominant_oracle_label(cluster_id: int, labels: np.ndarray, oracle_cluster_ids: np.ndarray) -> str:
    """Deterministic fake-annotator rule, derived from oracle ground truth (not random): a
    cluster whose members' most common oracle_cluster_id is a real action cluster (>= 0) is
    "task-definitive"; a cluster dominated by background windows (oracle id -1) is
    "exo-ambiguous"."""
    member_oracle_ids = oracle_cluster_ids[labels == cluster_id]
    dominant_id, _ = Counter(member_oracle_ids.tolist()).most_common(1)[0]
    return "task-definitive" if dominant_id != -1 else "exo-ambiguous"


def _fast_config(**overrides) -> PrototypingConfig:
    defaults = dict(n_clusters_per_round=15, faiss_niter=50, faiss_nredo=2, faiss_seed=47)
    defaults.update(overrides)
    return PrototypingConfig(**defaults)


def test_probe_rendering_pipeline_end_to_end(output_dir):
    config = _fast_config()
    result = _run_pipeline(config)

    participants = result["participants"]
    participant_ids = result["participant_ids"]
    oracle_cluster_ids = result["oracle_cluster_ids"]
    local_indices = result["local_indices"]
    centroids = result["centroids"]
    labels = result["labels"]
    distances = result["distances"]

    # --- fit_cosine_kmeans / assign_to_centroids sanity ---
    assert centroids.shape == (config.n_clusters_per_round, _SYNTH_CONFIG.embedding_dim)
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)
    assert np.all(distances >= 0.0) and np.all(distances <= 2.0)

    present_clusters = sorted(np.unique(labels).tolist())
    assert len(present_clusters) >= 2, "sane clustering: not everything collapsed into one cluster"

    # --- compute_cluster_stats for every cluster id proposed by k-means ---
    stats_by_cluster = {
        cluster_id: compute_cluster_stats(cluster_id, labels, distances, participant_ids)
        for cluster_id in range(config.n_clusters_per_round)
    }
    for cluster_id, stats in stats_by_cluster.items():
        if cluster_id in present_clusters:
            assert stats.n_embeddings > 0
        else:
            assert stats.n_embeddings == 0

    # --- render_probe_image (nearest + furthest) for at least 2 clusters ---
    frame_source = _PooledFrameSourceAdapter(SyntheticFrameSource(participants), local_indices)
    clusters_to_render = present_clusters[:2]
    assert len(clusters_to_render) == 2

    rendered = {}
    for cluster_id in clusters_to_render:
        nearest = sample_probe_indices(
            labels, distances, participant_ids, cluster_id,
            n_neighbors=config.n_probe_neighbors, side="nearest",
            max_participants=config.n_probe_participants_target,
        )
        furthest = sample_probe_indices(
            labels, distances, participant_ids, cluster_id,
            n_neighbors=config.n_probe_neighbors, side="furthest",
            max_participants=config.n_probe_participants_target,
        )
        assert len(nearest) > 0
        assert len(furthest) > 0
        # nearest/furthest blocks are visibly separated: each side's first pick (before any
        # participant-diversity fallback can apply, since `used_participants` starts empty)
        # is exactly the cluster's globally closest / globally furthest member -- a robust
        # invariant even for small clusters where the two blocks' *interiors* may overlap
        # once diversity-driven reordering kicks in.
        member_distances = distances[labels == cluster_id]
        assert nearest[0].distance == member_distances.min()
        assert furthest[0].distance == member_distances.max()
        assert nearest[0].distance <= furthest[0].distance

        out_path = output_dir / f"cluster_{cluster_id:03d}_probe_image.png"
        result_path = render_probe_image(
            cluster_id=cluster_id,
            nearest=nearest,
            furthest=furthest,
            stats=stats_by_cluster[cluster_id],
            frame_source=frame_source,
            out_path=out_path,
            ncols=config.probe_image_ncols,
        )
        rendered[cluster_id] = result_path

        assert result_path.exists()
        image = Image.open(result_path)
        image.verify()  # confirms a well-formed PNG
        image = Image.open(result_path)  # re-open: verify() invalidates the handle
        pixels = np.array(image.convert("RGB"))
        assert pixels.std() > 0  # non-trivial content, not a blank/uniform image
        assert len(np.unique(pixels.reshape(-1, 3), axis=0)) > 1

    # --- fake_annotate: deterministic rule derived from oracle data ---
    rendered_paths = list(rendered.values())
    path_to_cluster = {path: cluster_id for cluster_id, path in rendered.items()}

    def rule(path):
        return _dominant_oracle_label(path_to_cluster[path], labels, oracle_cluster_ids)

    annotations = fake_annotate(rendered_paths, rule)
    assert len(annotations) == len(rendered_paths)

    # Build qualification records for every present cluster (not just the 2 rendered above),
    # using the same deterministic rule, so the round-trip check below covers the whole
    # k-means output.
    records = [
        QualificationRecord(
            cluster_id=cluster_id,
            round=0,
            cluster_type=_dominant_oracle_label(cluster_id, labels, oracle_cluster_ids),
            annotator_id="fake-annotator",
        )
        for cluster_id in present_clusters
    ]

    # --- save_qualification_mapping / load_qualification_mapping round-trip ---
    qual_path = output_dir / "qualification_mapping.parquet"
    save_qualification_mapping(records, qual_path)
    loaded = load_qualification_mapping(qual_path)

    assert len(loaded) == len(present_clusters)
    allowed_categories = set(PrototypingConfig().annotation_categories)
    assert set(loaded["cluster_type"]).issubset(allowed_categories)


def test_probe_rendering_pipeline_deterministic_centroids():
    config = _fast_config(faiss_seed=47)

    centroids_a = _run_pipeline(config)["centroids"]
    centroids_b = _run_pipeline(config)["centroids"]

    np.testing.assert_allclose(centroids_a, centroids_b)
