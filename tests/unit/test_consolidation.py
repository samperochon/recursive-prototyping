"""Unit tests for recursive_prototyping.prototyping.consolidation."""

import numpy as np
import pytest

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.clustering import l2_normalize
from recursive_prototyping.prototyping.consolidation import (
    ConsolidationResult,
    MembershipTree,
    assemble_final_vocabulary,
    compute_group_validity,
    consolidate_once,
    default_g_grid,
    hierarchical_cluster,
    pairwise_cosine_distance,
    run_two_stage_consolidation,
    select_best_g,
    silhouette_scores_over_cuts,
)


def _two_tight_well_separated_groups(
    dim: int = 8, seed: int = 47
) -> tuple[np.ndarray, np.ndarray]:
    """6 unit vectors: 3 near-identical to e_0, 3 near-identical to e_1 (orthogonal base
    directions), tightly clustered (noise scale=0.01) so G=2 is the unambiguous silhouette
    optimum -- structurally similar to test_clustering.py's 3-cluster construction, but with
    2 groups of 3.
    """
    rng = np.random.default_rng(seed)
    base_directions = np.eye(2, dim)

    points, true_labels = [], []
    for group_idx, direction in enumerate(base_directions):
        noise = rng.normal(scale=0.01, size=(3, dim))
        points.append(l2_normalize(direction + noise))
        true_labels.extend([group_idx] * 3)

    return np.vstack(points), np.array(true_labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# pairwise_cosine_distance / hierarchical_cluster / default_g_grid /
# silhouette_scores_over_cuts / select_best_g
# ---------------------------------------------------------------------------


def test_pairwise_cosine_distance_shape_diagonal_and_symmetry():
    X, _ = _two_tight_well_separated_groups()
    D = pairwise_cosine_distance(X)

    assert D.shape == (6, 6)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
    np.testing.assert_array_equal(D, D.T)  # exact symmetry, required by squareform's checks
    assert np.all(D >= 0.0) and np.all(D <= 2.0)


def test_default_g_grid_falls_back_below_the_floor():
    # Pools of _MIN_G or fewer items cannot honour the floor of 15, so the
    # grid falls back to the full [2, n_items - 1] range rather than coming back empty.
    assert default_g_grid(6) == [2, 3, 4, 5]
    assert default_g_grid(3) == [2]
    assert default_g_grid(2) == []
    assert default_g_grid(15) == list(range(2, 15))


def test_default_g_grid_starts_at_the_floor_above_it():
    # The floor is load-bearing: a grid
    # starting at 2 lets silhouette-argmax take the degenerate top-of-dendrogram cut.
    assert default_g_grid(16) == [15]
    assert default_g_grid(20)[0] == 15
    assert 2 not in default_g_grid(100)


def test_default_g_grid_subsamples_large_ranges():
    grid = default_g_grid(1000)
    assert len(grid) <= 60
    assert grid[0] == 15
    assert grid[-1] == 999
    assert grid == sorted(set(grid))  # ascending, deduplicated


def test_silhouette_scores_over_cuts_and_select_best_g_recover_two_groups():
    X, _ = _two_tight_well_separated_groups()
    D = pairwise_cosine_distance(X)
    Z = hierarchical_cluster(D, linkage_method="complete")
    grid = default_g_grid(X.shape[0])

    scores = silhouette_scores_over_cuts(D, Z, grid)
    assert set(scores.keys()) <= set(grid)
    assert all(2 <= g < X.shape[0] for g in scores)

    assert select_best_g(scores) == 2


def test_select_best_g_raises_on_empty_dict():
    with pytest.raises(ValueError):
        select_best_g({})


# ---------------------------------------------------------------------------
# consolidate_once
# ---------------------------------------------------------------------------


def test_consolidate_once_recovers_two_groups_and_renormalizes_centroids():
    X, true_labels = _two_tight_well_separated_groups()

    result = consolidate_once(X, linkage_method="complete")

    assert isinstance(result, ConsolidationResult)
    assert result.best_g == 2
    assert result.group_centroids.shape == (2, X.shape[1])
    np.testing.assert_allclose(np.linalg.norm(result.group_centroids, axis=1), 1.0, atol=1e-8)

    # The recovered 2-way partition matches the true 3+3 split (up to label permutation):
    # every point from a given true group must land in the same recovered group.
    for true_group in (0, 1):
        recovered = set(result.assignment[true_labels == true_group].tolist())
        assert len(recovered) == 1


def test_consolidate_once_handles_too_few_items_without_raising():
    # n_items=2: default_g_grid(2) is empty (no G in [2, n_items-1] to search), so
    # consolidate_once must fall back to "every item is its own group" instead of
    # propagating select_best_g's ValueError -- see consolidate_once's docstring.
    X = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))

    result = consolidate_once(X)

    assert result.silhouette_by_g == {}
    assert result.best_g == 2
    np.testing.assert_array_equal(result.assignment, [0, 1])
    np.testing.assert_allclose(result.group_centroids, X, atol=1e-8)


# ---------------------------------------------------------------------------
# compute_group_validity
# ---------------------------------------------------------------------------


def test_compute_group_validity_all_invalid_when_pass_degenerated_to_singletons():
    # consolidate_once's trivial fallback (n_items too small to search a real G>=2 cut):
    # silhouette_by_g is empty, best_g == n_items, every group is its own singleton --
    # this is exactly NOT "well-defined and non-degenerate", so
    # every group must be marked invalid regardless of size.
    result = ConsolidationResult(
        assignment=np.array([0, 1]),
        group_centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
        silhouette_by_g={},
        best_g=2,
    )

    assert compute_group_validity(result) == {0: False, 1: False}


def test_compute_group_validity_valid_pass_still_marks_singleton_groups_invalid():
    # A genuine, non-trivial silhouette-argmax pass (silhouette_by_g non-empty) where group 2
    # has only one member: "well-defined" (the pass succeeded) is necessary but not sufficient
    # -- group 2 is still "degenerate" (nothing was actually merged into it) and must be False,
    # while groups 0 (size 2) and 1 (size 3) are True.
    result = ConsolidationResult(
        assignment=np.array([0, 0, 1, 1, 1, 2]),
        group_centroids=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        silhouette_by_g={3: 0.5, 2: 0.4},
        best_g=3,
    )

    assert compute_group_validity(result) == {0: True, 1: True, 2: False}


def test_compute_group_validity_counts_RAW_members_not_direct_items():
    """Pass 2's ``assignment`` indexes PASS-1 GROUPS, not raw centroids.

    A pass-2 group holding a single pass-1 group that itself holds many raw accepted centroids
    is NOT degenerate -- it represents all of those centroids -- but a direct member count sees
    "1 item" and demotes it out of tier 1. ``item_raw_weights`` is what carries the raw counts
    down; this pins that it is actually used: counting with ``np.bincount(assignment)`` fails.
    """
    result = ConsolidationResult(
        assignment=np.array([0, 1]),  # two pass-1 groups, one per pass-2 group
        group_centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
        silhouette_by_g={2: 0.5},
        best_g=2,
    )

    # Direct counts: both groups look like singletons.
    assert compute_group_validity(result) == {0: False, 1: False}

    # Weighted by raw membership: group 0 stands for 20 raw centroids, group 1 for exactly 1.
    assert compute_group_validity(result, item_raw_weights=np.array([20, 1])) == {
        0: True,
        1: False,
    }


def _many_tight_pairs(n_groups: int = 20, per_group: int = 2, dim: int = 32, seed: int = 47):
    """A pool big enough to clear ``default_g_grid``'s floor of 15 in BOTH passes.

    ``_two_tight_well_separated_groups`` has only 6 items, so its pass 2 runs on 2 group
    centroids and degenerates -- it cannot exercise pass-2 validity at all. This gives 40 raw
    centroids in 20 tight pairs, so pass 1 recovers the 20 pairs and pass 2 has 20 items to cut.
    """
    rng = np.random.default_rng(seed)
    directions = l2_normalize(rng.normal(size=(n_groups, dim)))
    points = [
        l2_normalize(direction + rng.normal(scale=0.02, size=(per_group, dim)))
        for direction in directions
    ]
    return np.vstack(points)


def test_two_stage_consolidation_weights_pass2_validity_by_raw_membership():
    """Raw-membership weighting, end to end, on a pool where it changes the answer.

    Pass 1 recovers 20 pairs; pass 2 cuts those 20 group centroids into 15 groups. Counting
    pass 2's DIRECT items calls 10 of those 15 degenerate singletons, when each actually
    represents a pair of raw accepted centroids. Counting with ``np.bincount(assignment)``
    flips 10 verdicts and this test fails.
    """
    centroids = _many_tight_pairs()
    config = PrototypingConfig()

    pass1, pass2, _tree, _pass1_validity, pass2_validity = run_two_stage_consolidation(
        centroids, config
    )

    pass1_raw_counts = np.bincount(pass1.assignment, minlength=pass1.group_centroids.shape[0])
    assert pass1_raw_counts.sum() == centroids.shape[0], "weights must cover every raw centroid"
    assert pass1_raw_counts.min() >= 2, "fixture: every pass-1 group should hold a whole pair"

    weighted = compute_group_validity(pass2, item_raw_weights=pass1_raw_counts)
    unweighted = compute_group_validity(pass2)

    assert pass2_validity == weighted
    assert sum(weighted.values()) > sum(unweighted.values()), (
        "fixture does not distinguish the two counting rules -- this test would pass "
        "vacuously against a direct item count"
    )


# ---------------------------------------------------------------------------
# assemble_final_vocabulary -- the 3-tier fallback (highest-value test)
# ---------------------------------------------------------------------------


def test_assemble_final_vocabulary_three_tier_fallback_and_dedup():
    # 5 raw accepted centroids, hand-routed through a small MembershipTree so that all three
    # fallback tiers, AND tier-1 deduplication with a majority-vote tie-break, are exercised
    # in one pass.
    #
    #   k=0, k=1 -> pass1_group 0 -> pass2_group 0 (VALID)                -> TIER 1 (deduped)
    #   k=2      -> pass1_group 1 -> pass2_group 1 (invalid), g1=1 VALID  -> TIER 2
    #   k=3      -> pass1_group 2 -> pass2_group 2 (invalid), g1=2 invalid -> TIER 3 (raw, exact)
    #   k=4      -> pass1_group 3 -> pass2_group 2 (invalid), g1=3 VALID  -> TIER 2
    raw_accepted_centroids = np.array(
        [
            [1.0, 0.0, 0.0],  # k0
            [0.9, 0.1, 0.0],  # k1
            [0.0, 1.0, 0.0],  # k2
            [0.0, 0.0, 1.0],  # k3 -- must survive into the vocabulary EXACTLY unmodified
            [0.3, 0.3, 0.9],  # k4
        ]
    )
    # k0/k1 share a valid pass-2 group and carry DIFFERENT cluster_types (tied 1-1), so the
    # majority-vote tie-break (first-wins in ascending raw-index order) is also exercised here.
    raw_cluster_types = [
        "exo-definitive",  # k0
        "task-definitive",  # k1
        "exo-definitive",  # k2
        "task-definitive",  # k3
        "exo-definitive",  # k4
    ]

    tree = MembershipTree(
        raw_to_pass1=np.array([0, 0, 1, 2, 3]),
        pass1_to_pass2=np.array([0, 1, 2, 2]),
    )
    pass1_validity = {0: True, 1: True, 2: False, 3: True}
    pass2_validity = {0: True, 1: False, 2: False}

    pass1 = ConsolidationResult(
        assignment=np.array([0, 0, 1, 2, 3]),
        group_centroids=np.array(
            [
                [0.99, 0.05, 0.0],  # group 0 (superseded by tier-1 for k0/k1, unused in output)
                [0.1, 0.99, 0.0],  # group 1 -> expected TIER-2 entry for k2
                [0.0, 0.1, 0.99],  # group 2 (unused -- k3 falls to tier 3)
                [0.2, 0.2, 0.95],  # group 3 -> expected TIER-2 entry for k4
            ]
        ),
        silhouette_by_g={4: 0.42},
        best_g=4,
    )
    pass2 = ConsolidationResult(
        assignment=np.array([0, 1, 2, 2]),
        group_centroids=np.array(
            [
                [0.7, 0.7, 0.0],  # group 0 -> expected TIER-1 entry for k0 & k1
                [0.0, 0.6, 0.8],  # group 1 (invalid, unused)
                [0.5, 0.0, 0.87],  # group 2 (invalid, unused)
            ]
        ),
        silhouette_by_g={3: 0.30},
        best_g=3,
    )

    vocab = assemble_final_vocabulary(
        tree, pass1_validity, pass2_validity, pass1, pass2,
        raw_accepted_centroids, raw_cluster_types,
    )

    assert vocab.centroids.shape == (4, 3)
    assert vocab.source_raw_indices == [[0, 1], [2], [4], [3]]
    assert vocab.cluster_types == [
        "exo-definitive",  # tier-1 dedup of k0/k1: tie 1-1, first-wins -> k0's type
        "exo-definitive",  # tier-2, k2's own type
        "exo-definitive",  # tier-2, k4's own type
        "task-definitive",  # tier-3, k3's own type
    ]

    np.testing.assert_allclose(vocab.centroids[0], [0.7, 0.7, 0.0])  # TIER 1 (deduped)
    np.testing.assert_allclose(vocab.centroids[1], [0.1, 0.99, 0.0])  # TIER 2
    np.testing.assert_allclose(vocab.centroids[2], [0.2, 0.2, 0.95])  # TIER 2
    # TIER 3: exact, unmodified array equality with the original raw centroid.
    np.testing.assert_array_equal(vocab.centroids[3], raw_accepted_centroids[3])

    # The reduction mapping symbolization uses: k0, k1 -> entry 0; k2 -> 1; k4 -> 2; k3 -> 3.
    np.testing.assert_array_equal(vocab.raw_to_prototype, [0, 0, 1, 3, 2])
    np.testing.assert_array_equal(vocab.raw_centroids, raw_accepted_centroids)


# ---------------------------------------------------------------------------
# run_two_stage_consolidation -- end-to-end wiring
# ---------------------------------------------------------------------------


def test_run_two_stage_consolidation_wires_passes_together():
    X, true_labels = _two_tight_well_separated_groups()
    config = PrototypingConfig()

    pass1, pass2, tree, pass1_validity, pass2_validity = run_two_stage_consolidation(X, config)

    n_pass1_groups = pass1.group_centroids.shape[0]
    n_pass2_groups = pass2.group_centroids.shape[0]
    assert tree.raw_to_pass1.shape == (X.shape[0],)
    assert tree.pass1_to_pass2.shape == (n_pass1_groups,)
    assert set(pass1_validity) == set(range(n_pass1_groups))
    assert set(pass2_validity) == set(range(n_pass2_groups))

    # pass1 finds G*=2, each group has 3 (>=2) members -- well-defined and non-degenerate:
    # both groups are valid.
    assert n_pass1_groups == 2
    assert all(pass1_validity.values())

    # pass2 runs on only the 2 pass-1 group centroids -- too few to search a real G>=2 cut
    # (default_g_grid(2) is empty), so consolidate_once falls back to its trivial singleton
    # case and every pass-2 group is invalid.
    assert n_pass2_groups == 2
    assert not any(pass2_validity.values())

    vocab = assemble_final_vocabulary(
        tree, pass1_validity, pass2_validity, pass1, pass2, X,
        ["task-definitive"] * X.shape[0],
    )

    # Every raw centroid is accounted for exactly once across the final vocabulary.
    all_source_indices = sorted(idx for group in vocab.source_raw_indices for idx in group)
    assert all_source_indices == list(range(X.shape[0]))
    assert vocab.centroids.shape[1] == X.shape[1]
    # Tier-2 fallback (pass1-valid, pass2-invalid): the final vocabulary is exactly the 2
    # pass-1 groups.
    assert vocab.centroids.shape[0] == 2


def test_run_two_stage_consolidation_rejects_non_cosine_metric():
    X, _ = _two_tight_well_separated_groups()
    config = PrototypingConfig(consolidation_metric="euclidean")

    with pytest.raises(NotImplementedError):
        run_two_stage_consolidation(X, config)
