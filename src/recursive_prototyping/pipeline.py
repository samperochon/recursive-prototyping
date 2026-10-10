"""End-to-end pipeline: vocabulary -> segmentation -> symbolization, plus persistence.

Wiring only: prototyping builds the vocabulary from all participants' pooled windows,
segmentation calibrates the penalty and segments, symbolization assigns the symbols. No
algorithm lives here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recursive_prototyping.config.io import save_config_snapshot
from recursive_prototyping.config.schema import (
    PrototypingConfig,
    SegmentationConfig,
    SymbolizationConfig,
)
from recursive_prototyping.io.arrays import save_array, save_arrays
from recursive_prototyping.prototyping.consolidation import (
    assemble_final_vocabulary,
    run_two_stage_consolidation,
)
from recursive_prototyping.prototyping.refinement import (
    AnnotateFn,
    run_recursive_loop,
    save_refinement_state,
)
from recursive_prototyping.prototyping.vpm import FrameSource
from recursive_prototyping.segmentation.penalty import (
    DEFAULT_HOP_SECONDS,
    calibrate_and_segment,
)
from recursive_prototyping.symbolization.assign import (
    assign_to_prototypes,
    background_filter,
    compute_prototype_thresholds,
)
from recursive_prototyping.symbolization.segments import expand_to_windows, symbolize


@dataclass
class ParticipantPipelineResult:
    """One participant's segmentation and symbolization outputs."""

    penalty: float  # -2 * n_windows * slope
    gamma: float | None  # median-heuristic rbf gamma; None for the linear kernel
    segmentation_cpts: np.ndarray  # change points [0, t1, ..., T] before merging, int64
    cpts: np.ndarray  # change points after short-segment merging; pairs with `symbols`
    symbols: np.ndarray  # int64, one per segment, in {-1} U [0, G)
    window_symbols: np.ndarray  # int64 [T]: `symbols` expanded to one per window


@dataclass
class PipelineResult:
    """Output of ``run_full_pipeline``."""

    vocabulary: np.ndarray  # [G, D] prototype means, unit-norm rows
    vocabulary_cluster_types: list[str]  # "task-definitive"/"exo-definitive" per prototype
    accepted_centroids: np.ndarray  # [K, D] centroids the windows are assigned against
    centroid_to_prototype: np.ndarray  # [K] centroid index -> row of `vocabulary`
    participants: dict[str, ParticipantPipelineResult]
    artifact_paths: dict[str, Path]  # artifact name -> path under out_dir


def _pool_embeddings(
    embeddings: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-participant sequences into (X_train, participant_ids, local_indices)."""
    xs, pids, locs = [], [], []
    for participant_id, X in embeddings.items():
        xs.append(X)
        pids.append(np.full(len(X), participant_id))
        locs.append(np.arange(len(X)))
    return np.concatenate(xs, axis=0), np.concatenate(pids), np.concatenate(locs)


def symbolize_participant(
    X: np.ndarray,
    centroids: np.ndarray,
    cpts: np.ndarray,
    config: SymbolizationConfig,
    thresholds: dict[int, float] | None = None,
    centroid_to_prototype: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symbolization for one participant; returns (filtered_labels, final_cpts, final_symbols).

    ``thresholds`` are the cohort-pooled radii from ``pooled_prototype_thresholds``; if
    omitted they come from this participant alone, which is not the paper's scope.
    """
    labels, distances = assign_to_prototypes(X, centroids, centroid_to_prototype)
    filtered = background_filter(
        labels, distances, n_sigma=config.background_n_sigma, thresholds=thresholds
    )
    final_cpts, final_symbols = symbolize(filtered, cpts, config)
    return filtered, final_cpts, final_symbols


def pooled_prototype_thresholds(
    embeddings: Mapping[str, np.ndarray],
    centroids: np.ndarray,
    config: SymbolizationConfig,
    centroid_to_prototype: np.ndarray | None = None,
) -> dict[int, float]:
    """Per-prototype radii m_g + n_sigma * sigma_g pooled over the cohort (paper Section 3.3)."""
    all_labels, all_distances = [], []
    for X in embeddings.values():
        labels, distances = assign_to_prototypes(X, centroids, centroid_to_prototype)
        all_labels.append(labels)
        all_distances.append(distances)
    return compute_prototype_thresholds(
        np.concatenate(all_labels),
        np.concatenate(all_distances),
        n_sigma=config.background_n_sigma,
    )


def run_full_pipeline(
    embeddings: Mapping[str, np.ndarray],
    frame_source: FrameSource,
    annotate_fn: AnnotateFn,
    prototyping_config: PrototypingConfig,
    segmentation_config: SegmentationConfig,
    symbolization_config: SymbolizationConfig,
    out_dir: str | Path,
    *,
    annotator_id: str,
    hop_seconds: float | Mapping[str, float] = DEFAULT_HOP_SECONDS,
) -> PipelineResult:
    """Run Modules 1 -> 2 -> 3 on a cohort and persist the results.

    ``embeddings`` maps participant_id -> [T_p, D]. ``hop_seconds`` is the seconds per
    embedding index, one float or a mapping per participant.

    Written under ``out_dir``: ``refinement/`` (probes, qualification tables, loop state),
    ``vocabulary.npy`` (prototype means), ``accepted_centroids.npy``,
    ``centroid_to_prototype.npy``, ``symbols.npz`` (keys ``{participant_id}__{field}`` for
    the four arrays of ``ParticipantPipelineResult``), ``calibration_summary.json`` and one
    JSON snapshot per config. Raises ValueError if the loop accepts no prototype.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prototyping: vocabulary
    X_train, participant_ids, local_indices = _pool_embeddings(embeddings)
    refinement_dir = out_dir / "refinement"
    state = run_recursive_loop(
        X_train=X_train,
        participant_ids=participant_ids,
        local_indices=local_indices,
        config=prototyping_config,
        annotate_fn=annotate_fn,
        base_frame_source=frame_source,
        annotator_id=annotator_id,
        out_dir=refinement_dir,
    )
    if state.accepted_centroids.shape[0] == 0:
        raise ValueError(
            "run_recursive_loop accepted zero prototypes -- nothing to consolidate into "
            "a vocabulary; check annotate_fn and PrototypingConfig."
        )
    save_refinement_state(state, refinement_dir)

    pass1, pass2, tree, pass1_validity, pass2_validity = run_two_stage_consolidation(
        state.accepted_centroids, prototyping_config
    )
    vocabulary = assemble_final_vocabulary(
        tree, pass1_validity, pass2_validity, pass1, pass2,
        state.accepted_centroids, state.accepted_cluster_types,
    )

    # Segmentation: penalty calibration, then change points
    calibration = calibrate_and_segment(embeddings, segmentation_config, hop_seconds)

    # Symbolization, with radii pooled over the cohort
    thresholds = pooled_prototype_thresholds(
        embeddings, vocabulary.raw_centroids, symbolization_config, vocabulary.raw_to_prototype
    )
    participants: dict[str, ParticipantPipelineResult] = {}
    for participant_id, X in embeddings.items():
        record = calibration.participants[participant_id]
        _filtered, final_cpts, final_symbols = symbolize_participant(
            X, vocabulary.raw_centroids, record.cpts, symbolization_config, thresholds,
            vocabulary.raw_to_prototype,
        )
        participants[participant_id] = ParticipantPipelineResult(
            penalty=record.penalty,
            gamma=record.gamma,
            segmentation_cpts=record.cpts,
            cpts=final_cpts,
            symbols=final_symbols,
            window_symbols=expand_to_windows(final_cpts, final_symbols),
        )

    # Persistence
    artifact_paths = {
        "refinement_state": refinement_dir,
        "vocabulary": out_dir / "vocabulary.npy",
        "accepted_centroids": out_dir / "accepted_centroids.npy",
        "centroid_to_prototype": out_dir / "centroid_to_prototype.npy",
        "symbols": out_dir / "symbols.npz",
        "calibration_summary": out_dir / "calibration_summary.json",
        "prototyping_config": out_dir / "prototyping_config.json",
        "segmentation_config": out_dir / "segmentation_config.json",
        "symbolization_config": out_dir / "symbolization_config.json",
    }
    save_array(artifact_paths["vocabulary"], vocabulary.centroids)
    save_array(artifact_paths["accepted_centroids"], vocabulary.raw_centroids)
    save_array(artifact_paths["centroid_to_prototype"], vocabulary.raw_to_prototype)
    per_participant_arrays: dict[str, np.ndarray] = {}
    for participant_id, result in participants.items():
        per_participant_arrays[f"{participant_id}__segmentation_cpts"] = result.segmentation_cpts
        per_participant_arrays[f"{participant_id}__cpts"] = result.cpts
        per_participant_arrays[f"{participant_id}__symbols"] = result.symbols
        per_participant_arrays[f"{participant_id}__window_symbols"] = result.window_symbols
    save_arrays(artifact_paths["symbols"], **per_participant_arrays)

    calibration_summary = {
        "slope": calibration.fit.slope,
        "r_squared": calibration.fit.r_squared,
        "participants": {
            participant_id: {
                "penalty": participants[participant_id].penalty,
                "gamma": participants[participant_id].gamma,
                "n_windows": int(embeddings[participant_id].shape[0]),
            }
            for participant_id in embeddings
        },
    }
    with artifact_paths["calibration_summary"].open("w") as f:
        json.dump(calibration_summary, f, indent=2, sort_keys=True)
        f.write("\n")

    save_config_snapshot(prototyping_config, artifact_paths["prototyping_config"])
    save_config_snapshot(segmentation_config, artifact_paths["segmentation_config"])
    save_config_snapshot(symbolization_config, artifact_paths["symbolization_config"])

    return PipelineResult(
        vocabulary=vocabulary.centroids,
        vocabulary_cluster_types=vocabulary.cluster_types,
        accepted_centroids=vocabulary.raw_centroids,
        centroid_to_prototype=vocabulary.raw_to_prototype,
        participants=participants,
        artifact_paths=artifact_paths,
    )
