"""Unit tests for recursive_prototyping.prototyping.vpm."""

import numpy as np
from PIL import Image

from recursive_prototyping.prototyping.vpm import (
    ClusterStats,
    ProbeSample,
    TILE_SIZE,
    compute_cluster_stats,
    render_probe_image,
    sample_probe_indices,
)


class _StubFrameSource:
    """Hand-built FrameSource returning a solid-color 8x8x3 uint8 frame per sample,
    so render_probe_image can be tested without depending on the real frame renderer."""

    def central_frame(self, participant_id: str, embedding_index: int) -> np.ndarray:
        seed = (hash(participant_id) % 200 + embedding_index * 17) % 256
        color = np.array([seed, (seed * 7) % 256, (seed * 13 + 40) % 256], dtype=np.uint8)
        return np.tile(color, (8, 8, 1))


# ---------------------------------------------------------------------------
# sample_probe_indices
# ---------------------------------------------------------------------------


def test_sample_probe_indices_nearest_known_order():
    labels = np.array([0, 0, 0, 1, 0])
    distances = np.array([0.5, 0.2, 0.8, 0.1, 0.4])
    participant_ids = np.array(["p1", "p2", "p3", "px", "p4"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0, n_neighbors=4, side="nearest"
    )

    expected = [
        ProbeSample("p2", 1, 0.2),
        ProbeSample("p4", 4, 0.4),
        ProbeSample("p1", 0, 0.5),
        ProbeSample("p3", 2, 0.8),
    ]
    assert result == expected


def test_sample_probe_indices_furthest_known_order():
    labels = np.array([0, 0, 0, 1, 0])
    distances = np.array([0.5, 0.2, 0.8, 0.1, 0.4])
    participant_ids = np.array(["p1", "p2", "p3", "px", "p4"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0, n_neighbors=4, side="furthest"
    )

    expected = [
        ProbeSample("p3", 2, 0.8),
        ProbeSample("p1", 0, 0.5),
        ProbeSample("p4", 4, 0.4),
        ProbeSample("p2", 1, 0.2),
    ]
    assert result == expected


def test_sample_probe_indices_diversity_cap_no_repeats_when_enough_participants():
    # p1 appears twice in the raw member list, but there are 5 distinct participants
    # available, so the n_neighbors=3 result must contain no repeated participant.
    labels = np.array([0, 0, 0, 0, 0, 0])
    distances = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    participant_ids = np.array(["p1", "p1", "p2", "p3", "p4", "p5"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0, n_neighbors=3, side="nearest"
    )

    expected = [
        ProbeSample("p1", 0, 0.10),
        ProbeSample("p2", 2, 0.20),
        ProbeSample("p3", 3, 0.30),
    ]
    assert result == expected
    pids = [s.participant_id for s in result]
    assert len(set(pids)) == len(pids) == 3


def test_sample_probe_indices_no_repeats_when_too_few_participants():
    # Only 2 distinct participants (p1, p2) but n_neighbors=4 requested, with 5 total
    # members available -- the result holds one sample per participant and falls short.
    labels = np.array([0, 0, 0, 0, 0])
    distances = np.array([0.10, 0.20, 0.30, 0.15, 0.25])
    participant_ids = np.array(["p1", "p1", "p1", "p2", "p2"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0, n_neighbors=4, side="nearest"
    )

    assert result == [ProbeSample("p1", 0, 0.10), ProbeSample("p2", 3, 0.15)]


def test_sample_probe_indices_never_exceeds_total_member_count():
    # Only 3 total members in the cluster; requesting more than that must not fabricate
    # extra (repeated) samples beyond the cluster's actual size.
    labels = np.array([5, 5, 5, 1, 1])
    distances = np.array([0.3, 0.1, 0.2, 0.05, 0.05])
    participant_ids = np.array(["a", "b", "c", "x", "y"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=5, n_neighbors=10, side="nearest"
    )

    assert len(result) == 3
    assert [s.participant_id for s in result] == ["b", "c", "a"]


def test_sample_probe_indices_max_participants_caps_diversity():
    # 4 distinct participants available and n_neighbors=4, but max_participants=2 stops the
    # selection after 2 distinct participants (p1, p2); p3/p4 are never introduced and the
    # further members of p1/p2 are not used to fill the remaining slots.
    labels = np.array([0, 0, 0, 0, 0, 0])
    distances = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    participant_ids = np.array(["p1", "p1", "p2", "p2", "p3", "p4"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0,
        n_neighbors=4, side="nearest", max_participants=2,
    )

    assert [s.participant_id for s in result] == ["p1", "p2"]
    assert [s.embedding_index for s in result] == [0, 2]


def test_sample_probe_indices_max_participants_enforced_even_if_it_falls_short():
    # 5 distinct participants, each with exactly ONE member -- max_participants=2 caps
    # diversity to 2. The cap is enforced (not merely a target), so the result falls short of
    # n_neighbors rather than pulling in a 3rd participant to hit the count.
    labels = np.array([0, 0, 0, 0, 0])
    distances = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    participant_ids = np.array(["p1", "p2", "p3", "p4", "p5"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0,
        n_neighbors=5, side="nearest", max_participants=2,
    )

    assert len(result) == 2
    pids = [s.participant_id for s in result]
    assert set(pids) == {"p1", "p2"}


def test_sample_probe_indices_max_participants_none_defaults_to_n_neighbors():
    # max_participants omitted entirely: the diversity cap is implicitly equal to
    # n_neighbors.
    labels = np.array([0, 0, 0, 0, 0, 0])
    distances = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    participant_ids = np.array(["p1", "p1", "p2", "p3", "p4", "p5"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0, n_neighbors=3, side="nearest"
    )

    pids = [s.participant_id for s in result]
    assert len(set(pids)) == len(pids) == 3


def test_sample_probe_indices_empty_cluster_returns_empty_list():
    labels = np.array([1, 1, 1])
    distances = np.array([0.1, 0.2, 0.3])
    participant_ids = np.array(["a", "b", "c"])

    result = sample_probe_indices(
        labels, distances, participant_ids, cluster_id=0, n_neighbors=5, side="nearest"
    )
    assert result == []


# ---------------------------------------------------------------------------
# compute_cluster_stats
# ---------------------------------------------------------------------------


def test_compute_cluster_stats_counts_and_mean_distance():
    # 10 members of cluster 0 (7 distinct participants, with repeats), plus 2 unrelated
    # rows from a different cluster that must be excluded from every statistic.
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    distances = np.array(
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.99]
    )
    participant_ids = np.array(
        ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p5", "p6", "p7", "px", "py"]
    )

    stats = compute_cluster_stats(0, labels, distances, participant_ids)

    assert isinstance(stats, ClusterStats)
    assert stats.cluster_id == 0
    assert stats.n_embeddings == 10
    assert stats.embedding_fraction == 10 / 12
    assert stats.n_participants == 7
    manual_mean = sum([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) / 10
    assert stats.mean_assignment_distance == manual_mean
    assert stats.avg_segment_occurrences is None
    assert stats.avg_segment_duration is None
    assert stats.normalized_timestamp_hist is None


def test_compute_cluster_stats_timestamp_histogram():
    labels = np.array([0] * 10 + [1] * 2)
    distances = np.zeros(12)
    participant_ids = np.array([f"p{i}" for i in range(12)])
    # 10 cluster-0 timestamps chosen to land in known bins of [0,1) split into 10 bins:
    # bin0 gets two (0.05, 0.05), bins 1-7 get one each, bin 8 gets none, bin 9 gets one.
    timestamps = np.array(
        [0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.95, 0.5, 0.5]
    )

    stats = compute_cluster_stats(
        0, labels, distances, participant_ids, timestamps=timestamps
    )

    assert stats.normalized_timestamp_hist is not None
    assert stats.normalized_timestamp_hist.sum() == 10
    expected_hist = np.array([2, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=np.float64)
    np.testing.assert_array_equal(stats.normalized_timestamp_hist, expected_hist)


def test_compute_cluster_stats_segment_fields():
    # Cluster 0 has members in segments s1 (len 3, both members from this cluster) and
    # s2 (len 4, only 1 member from this cluster, 3 from cluster 1).
    labels = np.array([0, 0, 1, 0, 1, 1, 1])
    distances = np.zeros(7)
    participant_ids = np.array(["p1", "p1", "p1", "p2", "p2", "p2", "p2"])
    segment_ids = np.array(["s1", "s1", "s1", "s2", "s2", "s2", "s2"])

    stats = compute_cluster_stats(
        0, labels, distances, participant_ids, segment_ids=segment_ids
    )

    # p1 has cluster-0 members only in segment s1 -> 1 distinct segment for p1.
    # p2 has cluster-0 members only in segment s2 -> 1 distinct segment for p2.
    assert stats.avg_segment_occurrences == 1.0
    # Cluster 0's segments are {s1 (length 3), s2 (length 4)} -> mean length 3.5.
    assert stats.avg_segment_duration == 3.5


# ---------------------------------------------------------------------------
# render_probe_image
# ---------------------------------------------------------------------------


def test_render_probe_image_produces_nonblank_png(output_dir):
    nearest = [
        ProbeSample("p1", 0, 0.1),
        ProbeSample("p2", 1, 0.2),
        ProbeSample("p3", 2, 0.3),
    ]
    furthest = [
        ProbeSample("p4", 3, 0.9),
        ProbeSample("p5", 4, 0.8),
        ProbeSample("p6", 5, 0.7),
    ]
    stats = ClusterStats(
        cluster_id=3,
        mean_assignment_distance=0.42,
        n_participants=6,
        n_embeddings=6,
        avg_segment_occurrences=1.5,
        avg_segment_duration=10.0,
        normalized_timestamp_hist=np.array([1, 0, 2, 0, 1, 0, 1, 0, 0, 1], dtype=np.float64),
    )

    out_path = output_dir / "cluster_003_probe_image.png"
    result_path = render_probe_image(
        cluster_id=3,
        nearest=nearest,
        furthest=furthest,
        stats=stats,
        frame_source=_StubFrameSource(),
        out_path=out_path,
        ncols=5,
    )

    assert result_path == out_path
    assert out_path.exists()

    image = Image.open(out_path)
    image.verify()  # confirms it's a well-formed PNG

    # Re-open after verify() (which invalidates the file handle for further use).
    image = Image.open(out_path)
    pixels = np.array(image.convert("RGB"))
    assert pixels.std() > 0  # not a uniform/blank image
    assert len(np.unique(pixels.reshape(-1, 3), axis=0)) > 1


def test_render_probe_image_creates_parent_dirs(output_dir):
    stats = ClusterStats(
        cluster_id=0,
        mean_assignment_distance=0.1,
        n_participants=1,
        n_embeddings=1,
        avg_segment_occurrences=None,
        avg_segment_duration=None,
        normalized_timestamp_hist=None,
    )
    out_path = output_dir / "nested" / "dir" / "probe.png"

    render_probe_image(
        cluster_id=0,
        nearest=[ProbeSample("p1", 0, 0.1)],
        furthest=[],
        stats=stats,
        frame_source=_StubFrameSource(),
        out_path=out_path,
    )

    assert out_path.exists()


def test_render_probe_image_tiles_blocks_and_overlays_timestamps(output_dir):
    stats = ClusterStats(0, 0.1, 3, 3, None, None, None)
    nearest = [ProbeSample("p1", 0, 0.1), ProbeSample("p2", 1, 0.2), ProbeSample("p3", 2, 0.3)]
    furthest = [ProbeSample("p4", 3, 0.9)]
    kwargs = dict(cluster_id=0, nearest=nearest, furthest=furthest, stats=stats,
                  frame_source=_StubFrameSource(), ncols=4)

    plain = np.array(Image.open(render_probe_image(out_path=output_dir / "plain.png", **kwargs)))
    stamped = np.array(Image.open(render_probe_image(
        out_path=output_dir / "stamped.png", timestamps=np.linspace(0, 1, 4), **kwargs
    )))

    # Two blocks of 2 columns side by side, 2 rows (3 nearest), no gap between tiles.
    tile_w, tile_h = TILE_SIZE
    mosaic = plain[plain.shape[0] - 2 * tile_h:]
    assert plain.shape[1] > 4 * tile_w and plain.shape[1] - 4 * tile_w < tile_w
    # Lower half of each tile, clear of the label in the top-right corner.
    first, second = mosaic[tile_h // 2:tile_h, :tile_w], mosaic[tile_h // 2:tile_h, tile_w:2 * tile_w]
    # solid stub frames fill their tile
    assert (first == first[0, 0]).all() and (second == second[0, 0]).all()
    assert not np.array_equal(first[0, 0], second[0, 0])
    assert (mosaic[tile_h:, tile_w:2 * tile_w] == 0).all()  # unused cell stays black

    assert not np.array_equal(plain, stamped)
