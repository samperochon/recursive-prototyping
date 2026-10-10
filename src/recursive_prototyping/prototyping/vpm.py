"""Visual Probing Mechanism (VPM): nearest/furthest probe sampling, cluster stats, and
the visual probe image a human annotator reviews for one candidate cluster.

Each image shows the samples nearest to and furthest from the centroid, spread over many
participants so that no single participant dominates.

``ProbeSample.embedding_index`` is the row's position in the arrays passed in; the caller's
``FrameSource`` must accept that index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

#: Probe tile (width, height) in pixels; frames are resized to it.
TILE_SIZE = (199, 112)


class FrameSource(Protocol):
    def central_frame(self, participant_id: str, embedding_index: int) -> np.ndarray:
        """Returns an HxWx3 uint8 image array for the given sample's representative frame."""


@dataclass
class ProbeSample:
    participant_id: str
    embedding_index: int
    distance: float


@dataclass
class ClusterStats:
    cluster_id: int
    mean_assignment_distance: float
    n_participants: int
    n_embeddings: int
    avg_segment_occurrences: float | None  # None if segment_ids not provided
    avg_segment_duration: float | None  # None if segment_ids not provided
    normalized_timestamp_hist: np.ndarray | None  # 10-bin histogram, None if timestamps not provided
    embedding_fraction: float | None = None  # share of the clustered rows in this cluster


def sample_probe_indices(
    labels: np.ndarray,
    distances: np.ndarray,
    participant_ids: np.ndarray,
    cluster_id: int,
    n_neighbors: int = 30,
    side: Literal["nearest", "furthest"] = "nearest",
    max_participants: int | None = None,
) -> list[ProbeSample]:
    """Pick up to ``n_neighbors`` members of ``cluster_id``, nearest (or furthest) first.

    Takes at most one sample per participant, from at most ``max_participants``
    participants (default ``n_neighbors``). The result is shorter than ``n_neighbors`` if
    the cluster has fewer participants. Samples are returned in selection order.
    """
    if side not in ("nearest", "furthest"):
        raise ValueError(f"side must be 'nearest' or 'furthest', got {side!r}")

    member_positions = np.flatnonzero(labels == cluster_id)
    if member_positions.size == 0:
        return []

    member_distances = distances[member_positions]
    sort_key = member_distances if side == "nearest" else -member_distances
    order = np.argsort(sort_key, kind="stable")
    sorted_positions = member_positions[order]

    n_target = min(n_neighbors, sorted_positions.size)
    participant_cap = n_neighbors if max_participants is None else max_participants

    # One sample per participant, up to the participant cap.
    selected: list[int] = []
    used_participants: set[str] = set()
    for pos in sorted_positions:
        if len(selected) >= n_target or len(used_participants) >= participant_cap:
            break
        pid = str(participant_ids[pos])
        if pid not in used_participants:
            selected.append(int(pos))
            used_participants.add(pid)

    return [
        ProbeSample(
            participant_id=str(participant_ids[pos]),
            embedding_index=int(pos),
            distance=float(distances[pos]),
        )
        for pos in selected
    ]


def compute_cluster_stats(
    cluster_id: int,
    labels: np.ndarray,
    distances: np.ndarray,
    participant_ids: np.ndarray,
    segment_ids: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> ClusterStats:
    """Summary statistics of one cluster, shown on its probe image.

    ``timestamps``, if given, must already be normalized to [0, 1]; they are binned into a
    10-bin histogram. ``segment_ids``, if given, yields the mean number of distinct segments
    per participant and the mean segment length, the length being counted over the full
    ``segment_ids`` array rather than the cluster's rows. Fields without input are None.
    """
    member_mask = labels == cluster_id
    n_embeddings = int(member_mask.sum())
    member_participants = participant_ids[member_mask]
    n_participants = int(np.unique(member_participants).size) if n_embeddings else 0

    if n_embeddings:
        mean_assignment_distance = float(distances[member_mask].mean())
    else:
        mean_assignment_distance = float("nan")

    avg_segment_occurrences: float | None = None
    avg_segment_duration: float | None = None
    if segment_ids is not None:
        if n_embeddings:
            member_segment_ids = segment_ids[member_mask]

            occurrence_counts = [
                int(np.unique(member_segment_ids[member_participants == pid]).size)
                for pid in np.unique(member_participants)
            ]
            avg_segment_occurrences = float(np.mean(occurrence_counts))

            distinct_segments = np.unique(member_segment_ids)
            all_segment_ids, all_counts = np.unique(segment_ids, return_counts=True)
            length_lookup = dict(zip(all_segment_ids, all_counts))
            durations = [length_lookup[seg] for seg in distinct_segments]
            avg_segment_duration = float(np.mean(durations))
        else:
            avg_segment_occurrences = float("nan")
            avg_segment_duration = float("nan")

    normalized_timestamp_hist: np.ndarray | None = None
    if timestamps is not None:
        member_timestamps = timestamps[member_mask]
        hist, _ = np.histogram(member_timestamps, bins=10, range=(0.0, 1.0))
        normalized_timestamp_hist = hist.astype(np.float64)

    return ClusterStats(
        cluster_id=cluster_id,
        mean_assignment_distance=mean_assignment_distance,
        n_participants=n_participants,
        n_embeddings=n_embeddings,
        avg_segment_occurrences=avg_segment_occurrences,
        avg_segment_duration=avg_segment_duration,
        normalized_timestamp_hist=normalized_timestamp_hist,
        embedding_fraction=n_embeddings / len(labels) if len(labels) else None,
    )


def render_probe_image(
    cluster_id: int,
    nearest: list[ProbeSample],
    furthest: list[ProbeSample],
    stats: ClusterStats,
    frame_source: FrameSource,
    out_path,
    ncols: int = 10,
    timestamps: np.ndarray | None = None,
) -> Path:
    """Save a PNG mosaic of the nearest (left block) and furthest (right block) frames,
    tiled without gaps, ``ncols`` tiles wide in total, under a one-line header of cluster
    stats. Returns ``out_path``.

    Each tile shows its distance to the centroid in the top-right corner, preceded by its
    normalized time when ``timestamps`` is given (values in [0, 1], indexed like
    ``ProbeSample.embedding_index``). The header's ``T%`` is the share of the cluster's
    windows in each tenth of the administration.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tile_w, tile_h = TILE_SIZE
    block_cols = max(1, ncols // 2)
    nrows = max(1, int(np.ceil(max(len(nearest), len(furthest)) / block_cols)))
    rule = 6
    header_h = 26

    block_w = block_cols * tile_w
    canvas = np.zeros((header_h + nrows * tile_h, 2 * block_w + rule, 3), dtype=np.uint8)
    canvas[:header_h] = 255
    canvas[:, block_w:block_w + rule] = 255

    labels: list[tuple[int, int, str]] = []
    for block, samples in enumerate((nearest, furthest)):
        x0 = block * (block_w + rule)
        for i, sample in enumerate(samples):
            row, col = divmod(i, block_cols)
            x, y = x0 + col * tile_w, header_h + row * tile_h
            frame = frame_source.central_frame(sample.participant_id, sample.embedding_index)
            tile = Image.fromarray(frame).convert("RGB").resize(TILE_SIZE, Image.BILINEAR)
            canvas[y:y + tile_h, x:x + tile_w] = np.asarray(tile)
            text = f"d={sample.distance:.2f}"
            if timestamps is not None:
                text = f"t={timestamps[sample.embedding_index]:.2f}  {text}"
            labels.append((x + tile_w - 3, y + 2, text))

    header_parts = [
        f"Cluster {stats.cluster_id}",
        f"d_avg={stats.mean_assignment_distance:.3f}",
        f"N={stats.n_participants} participants",
        f"N={stats.n_embeddings:,} windows"
        + (f" ({stats.embedding_fraction:.1%})" if stats.embedding_fraction is not None else ""),
    ]
    if stats.avg_segment_occurrences is not None:
        header_parts.append(f"segments/participant={stats.avg_segment_occurrences:.1f}")
    if stats.avg_segment_duration is not None:
        header_parts.append(f"segment length={stats.avg_segment_duration:.1f}")
    if stats.normalized_timestamp_hist is not None:
        hist = stats.normalized_timestamp_hist
        shares = 100 * hist / hist.sum() if hist.sum() else hist
        header_parts.append("T%=[" + " ".join(f"{v:.0f}" for v in shares) + "]")

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font_path = font_manager.findfont("DejaVu Sans")
    draw.text((6, header_h // 2), "   ".join(header_parts), fill="black", anchor="lm",
              font=ImageFont.truetype(font_path, 15))
    tile_font = ImageFont.truetype(font_path, 12)
    for x, y, text in labels:
        draw.text((x, y), text, fill="black", anchor="ra", font=tile_font,
                  stroke_width=2, stroke_fill="white")

    image.save(out_path, format="PNG")
    return out_path
