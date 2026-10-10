"""The recursive residual-set loop (paper Section 3.1).

Each round clusters the residual set, renders a probe image per candidate cluster, records
the annotator's verdicts and accepts the definitive centroids. The next residual set is
every row of an ambiguous cluster, plus the rows of accepted clusters whose cosine distance
to the centroid exceeds d_min; rows of noise clusters are dropped.

Clustering runs on a matrix pooled across participants, so a row position is a global
index. ``FrameSource.central_frame`` expects a participant-local window index;
``PooledFrameSourceAdapter`` translates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.io.arrays import load_array, load_arrays, save_array, save_arrays
from recursive_prototyping.io.tables import load_table, save_table
from recursive_prototyping.prototyping.annotation_io import (
    QualificationRecord,
    load_qualification_mapping,
    merge_qualification_mapping,
    save_qualification_mapping,
)
from recursive_prototyping.prototyping.clustering import assign_to_centroids, fit_cosine_kmeans
from recursive_prototyping.prototyping.vpm import (
    FrameSource,
    compute_cluster_stats,
    render_probe_image,
    sample_probe_indices,
)

# Definitive clusters are accepted, ambiguous ones are recycled whole, noise is dropped.
_ACCEPTED_CATEGORIES = {"task-definitive", "exo-definitive"}
_AMBIGUOUS_CATEGORIES = {"task-ambiguous", "exo-ambiguous"}

_QUALIFICATION_COLUMNS = ["cluster_id", "round", "cluster_type", "annotator_id"]

AnnotateFn = Callable[[list[Path], "tuple[str, ...]"], list[tuple[Path, str]]]


@dataclass
class RoundResult:
    round: int
    candidate_centroids: np.ndarray  # [C, D], all C clusters proposed this round
    cluster_types: list[str]  # length C, one of PrototypingConfig.annotation_categories, per candidate
    probe_paths: list[Path]  # length C, path to each candidate's rendered probe image


@dataclass
class PendingRound:
    """A round that is rendered but not yet annotated: what ``start_round`` hands to
    ``finish_round``. Persisted so a live annotation pass can resume after a kernel restart
    without re-running the clustering.
    """

    round: int
    residual_global_indices: np.ndarray  # [n_p] int, X_train rows forming this round's R_p
    centroids: np.ndarray  # [C, D] this round's candidate centroids
    labels: np.ndarray  # [n_p] each R_p row's assigned candidate cluster
    distances: np.ndarray  # [n_p] each R_p row's distance to that cluster's centroid
    probe_paths: list[Path]  # length C, the rendered probe image awaiting a verdict


@dataclass
class RefinementState:
    rounds: list[RoundResult]
    accepted_centroids: np.ndarray  # [K, D], all accepted centroids across all rounds so far, in acceptance order
    accepted_cluster_types: list[str]  # length K, "task-definitive" or "exo-definitive" per accepted centroid
    accepted_probe_paths: list[Path]  # length K, each accepted centroid's own probe image path
    qualification_mapping: pd.DataFrame  # accumulated across rounds, via annotation_io
    # Resumability. Defaulted so states without these fields still load.
    residual_mask: np.ndarray | None = None  # [len(X_train)] bool, rows still in the residual set
    next_round: int = 1  # 1-based number of the round ``start_round`` will render next
    pending: PendingRound | None = None  # set between start_round and finish_round, else None


def init_refinement_state(X_train: np.ndarray) -> RefinementState:
    """A fresh state for a run over ``X_train``: R_1 = all of X_train, nothing accepted yet."""
    return RefinementState(
        rounds=[],
        accepted_centroids=np.empty((0, X_train.shape[1]), dtype=X_train.dtype),
        accepted_cluster_types=[],
        accepted_probe_paths=[],
        qualification_mapping=pd.DataFrame(columns=_QUALIFICATION_COLUMNS),
        residual_mask=np.ones(len(X_train), dtype=bool),
        next_round=1,
        pending=None,
    )


class PooledFrameSourceAdapter:
    """Wraps a FrameSource so it accepts global row indices of a pooled multi-participant
    matrix, translating each to its participant-local window index.
    """

    def __init__(self, local_indices: np.ndarray, base: FrameSource) -> None:
        self.local_indices = local_indices  # local_indices[global_row] = local window index
        self.base = base

    def central_frame(self, participant_id: str, embedding_index: int) -> np.ndarray:
        local = int(self.local_indices[embedding_index])
        return self.base.central_frame(participant_id, local)


def run_generative_step(
    R_p: np.ndarray, config: PrototypingConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine k-means on ``R_p``, then assign every row to its nearest candidate centroid.

    Clusters are numbered by decreasing size, so cluster 0 is the largest and the annotator
    reviews the most populated candidates first.

    Returns (centroids [C, D], labels [len(R_p)], distances [len(R_p)]).
    """
    centroids, _kmeans = fit_cosine_kmeans(R_p, config.n_clusters_per_round, config)
    labels, distances = assign_to_centroids(R_p, centroids)

    order = np.argsort(-np.bincount(labels, minlength=centroids.shape[0]), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return centroids[order], rank[labels], distances


def split_by_qualification(
    qualification_df: pd.DataFrame, round_number: int, annotator_id: str
) -> tuple[list[int], list[int], list[int]]:
    """Split one round's cluster ids, for one annotator, into sorted
    (accepted_ids, ambiguous_ids, noise_ids). Ids index that round's candidate centroids.
    """
    subset = qualification_df[
        (qualification_df["round"] == round_number)
        & (qualification_df["annotator_id"] == annotator_id)
    ]

    accepted_ids = sorted(
        int(cid) for cid in subset.loc[subset["cluster_type"].isin(_ACCEPTED_CATEGORIES), "cluster_id"]
    )
    ambiguous_ids = sorted(
        int(cid) for cid in subset.loc[subset["cluster_type"].isin(_AMBIGUOUS_CATEGORIES), "cluster_id"]
    )
    noise_ids = sorted(
        int(cid) for cid in subset.loc[subset["cluster_type"] == "noise", "cluster_id"]
    )
    return accepted_ids, ambiguous_ids, noise_ids


def build_residual_set(
    labels: np.ndarray,
    distances: np.ndarray,
    accepted_ids: set[int],
    ambiguous_ids: set[int],
    d_min: float,
) -> np.ndarray:
    """Next round's residual set, as a boolean mask over the current R_p's rows.

    Per the paper (Section 3.1): a row is recycled if it is (i) assigned to an ambiguous
    cluster, or (ii) assigned to an accepted cluster with a cosine distance to that centroid
    above ``d_min``. Rows of noise clusters are dropped.
    """
    is_ambiguous = np.isin(labels, np.array(sorted(ambiguous_ids), dtype=labels.dtype))
    is_accepted = np.isin(labels, np.array(sorted(accepted_ids), dtype=labels.dtype))
    return is_ambiguous | (is_accepted & (distances > d_min))


def start_round(
    state: RefinementState,
    X_train: np.ndarray,
    participant_ids: np.ndarray,
    local_indices: np.ndarray,
    config: PrototypingConfig,
    base_frame_source: FrameSource,
    out_dir: str | Path,
) -> list[Path]:
    """Cluster the residual set and render one round's probe images, without annotating.

    The round is left in ``state.pending`` (``state`` is mutated in place); pass its verdicts
    to ``finish_round``.

    Returns the probe paths, or ``[]`` with ``state.pending`` left None when the loop is
    over: ``config.n_rounds`` rounds are done, ``config.max_accepted_prototypes`` is
    exceeded, or the residual set has fewer rows than ``config.n_clusters_per_round``. An
    empty return is the normal end of the loop, not an error.

    A residual with fewer than 39 rows per cluster is not refused: faiss warns on stderr and
    still returns centroids fitted to few points.

    Raises:
        ValueError: if a round is already pending, or if ``state`` has no ``residual_mask``
            (build states with ``init_refinement_state``).
    """
    if state.pending is not None:
        raise ValueError(
            f"round {state.pending.round} is already rendered and awaiting verdicts -- call "
            "finish_round with its annotations before starting another round."
        )
    if state.residual_mask is None:
        raise ValueError(
            "state.residual_mask is None: this state was not built by init_refinement_state "
            "(or was loaded from a state directory written before resumability existed), so "
            "there is no residual set to continue from."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (
        config.max_accepted_prototypes is not None
        and state.accepted_centroids.shape[0] > config.max_accepted_prototypes
    ):
        return []
    if state.next_round > config.n_rounds:
        return []

    residual_global_indices = np.flatnonzero(state.residual_mask)
    if residual_global_indices.size < config.n_clusters_per_round:
        # Fewer rows than clusters: faiss would raise, so end the loop instead.
        return []

    p = state.next_round
    R_p = X_train[residual_global_indices]
    participant_ids_p = participant_ids[residual_global_indices]
    local_indices_p = local_indices[residual_global_indices]
    # Normalized time of each window within its participant's sequence.
    _, inverse, counts = np.unique(participant_ids, return_inverse=True, return_counts=True)
    timestamps_p = (local_indices / counts[inverse])[residual_global_indices]

    centroids, labels, distances = run_generative_step(R_p, config)
    n_clusters = centroids.shape[0]

    # Probe samples are row positions in R_p; the adapter maps them to local window indices.
    frame_source = PooledFrameSourceAdapter(local_indices_p, base_frame_source)

    probe_paths: list[Path] = []
    for c in range(n_clusters):
        nearest = sample_probe_indices(
            labels, distances, participant_ids_p, cluster_id=c,
            n_neighbors=config.n_probe_neighbors, side="nearest",
            max_participants=config.n_probe_participants_target,
        )
        furthest = sample_probe_indices(
            labels, distances, participant_ids_p, cluster_id=c,
            n_neighbors=config.n_probe_neighbors, side="furthest",
            max_participants=config.n_probe_participants_target,
        )
        stats = compute_cluster_stats(
            c, labels, distances, participant_ids_p, timestamps=timestamps_p
        )

        out_path = out_dir / f"round_{p}_cluster_{c}.png"
        render_probe_image(
            cluster_id=c,
            nearest=nearest,
            furthest=furthest,
            stats=stats,
            frame_source=frame_source,
            out_path=out_path,
            ncols=config.probe_image_ncols,
            timestamps=timestamps_p,
        )
        probe_paths.append(out_path)

    state.pending = PendingRound(
        round=p,
        residual_global_indices=residual_global_indices,
        centroids=centroids,
        labels=labels,
        distances=distances,
        probe_paths=probe_paths,
    )
    return probe_paths


def finish_round(
    state: RefinementState,
    annotations: list[tuple[Path, str]],
    config: PrototypingConfig,
    annotator_id: str,
    out_dir: str | Path,
) -> RefinementState:
    """Apply one round's verdicts: record them, accept centroids, rebuild the residual set.

    ``annotations`` is an ``AnnotateFn`` result, e.g. from
    ``annotation_widget.collect_annotations``. ``state`` is mutated in place and returned,
    with ``state.pending`` cleared and ``state.next_round`` advanced.

    Raises:
        ValueError: if no round is pending.
        KeyError: if ``annotations`` has no verdict for a rendered probe path.
    """
    if state.pending is None:
        raise ValueError(
            "no round is pending -- call start_round to render a round's probe images before "
            "finishing it."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = state.pending
    p = pending.round
    n_clusters = pending.centroids.shape[0]

    # Key verdicts by path, not by return order, so a mismatch raises instead of mislabelling.
    label_by_path = {Path(path): label for path, label in annotations}
    cluster_types = [label_by_path[pending.probe_paths[c]] for c in range(n_clusters)]
    # A label outside the categories would fall in none of accepted / ambiguous / noise and
    # its rows would be dropped like noise, silently; a hand-edited checkpoint can do this.
    unknown = sorted({t for t in cluster_types if t not in config.annotation_categories})
    if unknown:
        raise ValueError(
            f"verdict(s) {unknown} are not in config.annotation_categories "
            f"{tuple(config.annotation_categories)}"
        )

    records = [
        QualificationRecord(
            cluster_id=c, round=p, cluster_type=cluster_types[c], annotator_id=annotator_id
        )
        for c in range(n_clusters)
    ]
    round_qual_path = out_dir / f"round_{p}_qualification.parquet"
    save_qualification_mapping(records, round_qual_path)
    round_df = load_qualification_mapping(round_qual_path)
    state.qualification_mapping = merge_qualification_mapping(state.qualification_mapping, round_df)

    accepted_ids, ambiguous_ids, _noise_ids = split_by_qualification(
        state.qualification_mapping, round_number=p, annotator_id=annotator_id
    )

    if accepted_ids:
        state.accepted_centroids = np.concatenate(
            [state.accepted_centroids, pending.centroids[accepted_ids]], axis=0
        )
        state.accepted_cluster_types.extend(cluster_types[c] for c in accepted_ids)
        state.accepted_probe_paths.extend(pending.probe_paths[c] for c in accepted_ids)

    residual_mask_local = build_residual_set(
        pending.labels,
        pending.distances,
        set(accepted_ids),
        set(ambiguous_ids),
        config.residual_distance_threshold,
    )
    next_residual_global = pending.residual_global_indices[residual_mask_local]
    residual_mask = np.zeros(len(state.residual_mask), dtype=bool)
    residual_mask[next_residual_global] = True
    state.residual_mask = residual_mask

    state.rounds.append(
        RoundResult(
            round=p,
            candidate_centroids=pending.centroids,
            cluster_types=cluster_types,
            probe_paths=pending.probe_paths,
        )
    )

    state.pending = None
    state.next_round = p + 1
    return state


def run_recursive_loop(
    X_train: np.ndarray,
    participant_ids: np.ndarray,
    local_indices: np.ndarray,
    config: PrototypingConfig,
    annotate_fn: AnnotateFn,
    base_frame_source: FrameSource,
    annotator_id: str,
    out_dir: str | Path,
) -> RefinementState:
    """Run every round with a callable annotator, looping ``start_round`` / ``finish_round``.

    ``X_train``, ``participant_ids`` and ``local_indices`` are parallel arrays pooled across
    participants; ``local_indices[i]`` is row i's window index within its own participant.

    A live pigeon pass cannot use this (``annotate_probe_images`` is not an ``AnnotateFn``):
    call the two functions directly, one notebook cell each, saving the state in between.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = init_refinement_state(X_train)
    while True:
        probe_paths = start_round(
            state, X_train, participant_ids, local_indices, config, base_frame_source, out_dir
        )
        if not probe_paths:
            break
        annotations = annotate_fn(probe_paths, config.annotation_categories)
        finish_round(state, annotations, config, annotator_id, out_dir)

    return state


def save_refinement_state(state: RefinementState, out_dir: str | Path) -> None:
    """Write the state to ``out_dir`` as npy/npz, parquet and json (no pickle).

    Layout:
      accepted_centroids.npy, qualification_mapping.parquet
      rounds/round_{p}_candidate_centroids.npy   one per finished round
      residual_mask.npy                          omitted when None
      pending.npz                                the pending round's arrays, omitted when none
      meta.json                                  everything else

    The residual mask and pending round let a run resume between the two annotation cells
    without re-running the clustering.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_array(out_dir / "accepted_centroids.npy", state.accepted_centroids)
    save_table(out_dir / "qualification_mapping.parquet", state.qualification_mapping)

    rounds_meta = []
    for round_result in state.rounds:
        centroids_rel_path = f"rounds/round_{round_result.round}_candidate_centroids.npy"
        save_array(out_dir / centroids_rel_path, round_result.candidate_centroids)
        rounds_meta.append(
            {
                "round": round_result.round,
                "cluster_types": round_result.cluster_types,
                "probe_paths": [str(p) for p in round_result.probe_paths],
                "candidate_centroids_path": centroids_rel_path,
            }
        )

    if state.residual_mask is not None:
        save_array(out_dir / "residual_mask.npy", state.residual_mask)

    pending_meta = None
    # Drop a pending.npz left by an earlier save so the directory agrees with meta.json.
    (out_dir / "pending.npz").unlink(missing_ok=True)
    if state.pending is not None:
        save_arrays(
            out_dir / "pending.npz",
            residual_global_indices=state.pending.residual_global_indices,
            centroids=state.pending.centroids,
            labels=state.pending.labels,
            distances=state.pending.distances,
        )
        pending_meta = {
            "round": state.pending.round,
            "probe_paths": [str(path) for path in state.pending.probe_paths],
        }

    meta = {
        "accepted_cluster_types": state.accepted_cluster_types,
        "accepted_probe_paths": [str(p) for p in state.accepted_probe_paths],
        "rounds": rounds_meta,
        "next_round": state.next_round,
        "pending": pending_meta,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def load_refinement_state(out_dir: str | Path) -> RefinementState:
    """Inverse of save_refinement_state."""
    out_dir = Path(out_dir)

    with (out_dir / "meta.json").open("r") as f:
        meta = json.load(f)

    accepted_centroids = load_array(out_dir / "accepted_centroids.npy")
    qualification_mapping = load_table(out_dir / "qualification_mapping.parquet")

    rounds = [
        RoundResult(
            round=round_meta["round"],
            candidate_centroids=load_array(out_dir / round_meta["candidate_centroids_path"]),
            cluster_types=round_meta["cluster_types"],
            probe_paths=[Path(p) for p in round_meta["probe_paths"]],
        )
        for round_meta in meta["rounds"]
    ]

    residual_mask_path = out_dir / "residual_mask.npy"
    residual_mask = load_array(residual_mask_path) if residual_mask_path.exists() else None

    pending_meta = meta.get("pending")
    pending = None
    if pending_meta is not None:
        arrays = load_arrays(out_dir / "pending.npz")
        pending = PendingRound(
            round=pending_meta["round"],
            residual_global_indices=arrays["residual_global_indices"],
            centroids=arrays["centroids"],
            labels=arrays["labels"],
            distances=arrays["distances"],
            probe_paths=[Path(path) for path in pending_meta["probe_paths"]],
        )

    return RefinementState(
        rounds=rounds,
        accepted_centroids=accepted_centroids,
        accepted_cluster_types=meta["accepted_cluster_types"],
        accepted_probe_paths=[Path(p) for p in meta["accepted_probe_paths"]],
        qualification_mapping=qualification_mapping,
        residual_mask=residual_mask,
        next_round=meta.get("next_round", 1),
        pending=pending,
    )
