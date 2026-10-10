"""Consolidate the accepted centroids into the final vocabulary.

The recursive loop can accept the same visual concept in several rounds. Two passes of
complete-linkage HAC on cosine distance, each cut at its silhouette-maximizing number of
groups, merge those duplicates. ``assemble_final_vocabulary`` then represents each accepted
centroid by its pass-2 group if valid, else its pass-1 group if valid, else itself, so none
is dropped.

Group validity has two sources. ``compute_group_validity`` is a headless proxy (a genuine
silhouette cut and at least 2 raw members); ``meta_prototypes.py`` provides the human
verdict the paper actually defines.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from recursive_prototyping.config.schema import PrototypingConfig
from recursive_prototyping.prototyping.clustering import l2_normalize

# The silhouette search starts at 15 groups when there are more than 15 items, which avoids
# the trivial 2-group optimum; the paper's formula has no floor.
_MIN_G = 15

# Above this many candidate group counts, the search grid is subsampled evenly.
_MAX_GRID_SIZE = 60


def pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    """[N, N] cosine distance matrix of the rows of ``X``, which are L2-normalized first.

    The result is made exactly symmetric and zero-diagonal, as ``squareform`` in
    ``hierarchical_cluster`` requires.
    """
    X_normalized = l2_normalize(np.asarray(X, dtype=np.float64))
    similarity = X_normalized @ X_normalized.T
    D = 1.0 - similarity
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, 2.0)


def hierarchical_cluster(D: np.ndarray, linkage_method: str = "complete") -> np.ndarray:
    """Linkage matrix of ``D``, which must be a symmetric, zero-diagonal distance matrix."""
    condensed = squareform(D, checks=True)
    return linkage(condensed, method=linkage_method)


def default_g_grid(n_items: int) -> list[int]:
    """Candidate group counts for the silhouette search: ``_MIN_G`` to ``n_items - 1``.

    The floor keeps silhouette-argmax from picking a degenerate two- or three-group cut.
    Pools of ``_MIN_G`` items or fewer use the full range 2 to ``n_items - 1``. A range
    longer than ``_MAX_GRID_SIZE`` is subsampled to evenly spaced values over the same span.
    """
    low = _MIN_G if n_items > _MIN_G else 2
    full_range = list(range(low, n_items))
    if len(full_range) <= _MAX_GRID_SIZE:
        return full_range
    subsampled = np.linspace(low, n_items - 1, _MAX_GRID_SIZE, dtype=int)
    return sorted(set(subsampled.tolist()))


def silhouette_scores_over_cuts(
    D: np.ndarray, linkage_matrix: np.ndarray, g_grid: list[int]
) -> dict[int, float]:
    """Silhouette score of the dendrogram cut at each ``g`` in ``g_grid``.

    Omits any ``g`` outside 2 to ``n_items - 1`` (silhouette is undefined there) and any
    ``g`` for which ties in the linkage make the cut yield fewer than ``g`` clusters.
    """
    n_items = D.shape[0]
    scores: dict[int, float] = {}
    for g in g_grid:
        if g < 2 or g >= n_items:
            continue
        labels = fcluster(linkage_matrix, t=g, criterion="maxclust")
        if len(np.unique(labels)) != g:
            continue
        scores[g] = float(silhouette_score(D, labels, metric="precomputed"))
    return scores


def select_best_g(silhouette_by_g: dict[int, float]) -> int:
    """The ``g`` with the highest score. Raises ValueError if the dict is empty."""
    if not silhouette_by_g:
        raise ValueError(
            "silhouette_by_g is empty -- no candidate G in the search grid produced a valid "
            "silhouette score to select from."
        )
    return max(silhouette_by_g, key=silhouette_by_g.get)


@dataclass
class ConsolidationResult:
    assignment: np.ndarray  # [n_items] -> group id in [0, G)
    group_centroids: np.ndarray  # [G, D]: mean of each group's members, L2-renormalized
    silhouette_by_g: dict[int, float]
    best_g: int


def consolidate_once(centroids: np.ndarray, linkage_method: str = "complete") -> ConsolidationResult:
    """One HAC + silhouette-argmax pass, used for both consolidation passes.

    Group centroids are the L2-renormalized means of their members. If no group count gives
    a valid silhouette score (two items or fewer, or every cut collapsed by ties), each item
    is returned as its own group with ``best_g = n_items``, so a small second pass does not
    fail.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    n_items = centroids.shape[0]

    D = pairwise_cosine_distance(centroids)
    linkage_matrix = hierarchical_cluster(D, linkage_method=linkage_method) if n_items >= 2 else None

    g_grid = default_g_grid(n_items)
    silhouette_by_g = (
        silhouette_scores_over_cuts(D, linkage_matrix, g_grid)
        if linkage_matrix is not None and g_grid
        else {}
    )

    if not silhouette_by_g:
        assignment = np.arange(n_items)
        group_centroids = l2_normalize(centroids)
        return ConsolidationResult(
            assignment=assignment,
            group_centroids=group_centroids,
            silhouette_by_g={},
            best_g=n_items,
        )

    best_g = select_best_g(silhouette_by_g)
    raw_labels = fcluster(linkage_matrix, t=best_g, criterion="maxclust")
    _, assignment = np.unique(raw_labels, return_inverse=True)
    assignment = assignment.astype(np.int64)

    n_groups = int(assignment.max()) + 1
    group_centroids = np.stack(
        [centroids[assignment == g].mean(axis=0) for g in range(n_groups)]
    )
    group_centroids = l2_normalize(group_centroids)

    return ConsolidationResult(
        assignment=assignment,
        group_centroids=group_centroids,
        silhouette_by_g=silhouette_by_g,
        best_g=best_g,
    )


def compute_group_validity(
    result: "ConsolidationResult", item_raw_weights: np.ndarray | None = None
) -> dict[int, bool]:
    """Headless proxy for per-group validity, used when no annotator is available.

    The paper defines validity as a human verdict on each group; ``meta_prototypes.py``
    collects it. Here a group is valid when the pass made a genuine silhouette cut
    (``result.silhouette_by_g`` is non-empty) and the group holds at least 2 raw accepted
    centroids. Under the paper's rule a singleton is valid; this proxy marks it invalid.

    ``item_raw_weights`` is the number of raw centroids under each item of this pass. Pass
    it for pass 2, whose items are pass-1 groups; None counts each item as 1.
    """
    n_groups = result.group_centroids.shape[0]
    if not result.silhouette_by_g:
        return {g: False for g in range(n_groups)}
    if item_raw_weights is None:
        item_raw_weights = np.ones(len(result.assignment), dtype=int)
    group_sizes = np.bincount(result.assignment, weights=item_raw_weights, minlength=n_groups)
    return {g: bool(group_sizes[g] >= 2) for g in range(n_groups)}


@dataclass
class MembershipTree:
    """Maps each raw accepted centroid to its pass-1 group, and each pass-1 group to its
    pass-2 group."""

    raw_to_pass1: np.ndarray  # [K] -> pass-1 group id in [0, G1)
    pass1_to_pass2: np.ndarray  # [G1] -> pass-2 group id in [0, G2)


def run_two_stage_consolidation(
    accepted_centroids: np.ndarray,
    config: PrototypingConfig,
) -> tuple["ConsolidationResult", "ConsolidationResult", "MembershipTree", dict, dict]:
    """Run both consolidation passes and score each with ``compute_group_validity``.

    Pass 2 clusters all pass-1 group centroids, valid or not.

    Returns (pass1, pass2, tree, pass1_validity, pass2_validity).

    Raises:
        NotImplementedError: if ``config.consolidation_metric`` is not "cosine".
    """
    if config.consolidation_metric != "cosine":
        raise NotImplementedError(
            f"consolidation_metric={config.consolidation_metric!r} is not supported -- this "
            "module implements only the alpha=1 (plain cosine-distance HAC) case."
        )

    accepted_centroids = np.asarray(accepted_centroids)

    pass1 = consolidate_once(accepted_centroids, config.consolidation_linkage)
    pass1_validity = compute_group_validity(pass1)

    pass2 = consolidate_once(pass1.group_centroids, config.consolidation_linkage)
    # Pass-2 items are pass-1 groups: weight each by its number of raw centroids.
    pass1_raw_counts = np.bincount(
        pass1.assignment, minlength=pass1.group_centroids.shape[0]
    )
    pass2_validity = compute_group_validity(pass2, item_raw_weights=pass1_raw_counts)

    tree = MembershipTree(raw_to_pass1=pass1.assignment, pass1_to_pass2=pass2.assignment)

    return pass1, pass2, tree, pass1_validity, pass2_validity


@dataclass
class Vocabulary:
    centroids: np.ndarray  # [G, D] final consolidated vocabulary
    cluster_types: list[str]  # "task-definitive"/"exo-definitive" per final vocabulary entry
    source_raw_indices: list[list[int]]  # per final vocabulary entry, contributing raw indices
    raw_centroids: np.ndarray  # [K, D] accepted centroids; symbolization assigns windows to these
    raw_to_prototype: np.ndarray  # [K] centroid -> prototype reduction mapping, for symbolization


def _majority_vote(labels: list[str]) -> str:
    """Most common label; ties go to the one appearing first in ``labels``."""
    counts = Counter(labels)
    best_count = max(counts.values())
    for label in labels:
        if counts[label] == best_count:
            return label
    raise AssertionError("unreachable: labels is non-empty")  # pragma: no cover


def assemble_final_vocabulary(
    tree: MembershipTree,
    pass1_validity: dict[int, bool],
    pass2_validity: dict[int, bool],
    pass1: ConsolidationResult,
    pass2: ConsolidationResult,
    raw_accepted_centroids: np.ndarray,
    raw_cluster_types: list[str],
) -> Vocabulary:
    """Build the final vocabulary with the 3-tier fallback.

    Each raw accepted centroid is represented by its pass-2 group centroid if that group is
    valid, else by its pass-1 group centroid if that group is valid, else by itself. Raw
    centroids sharing a group share one entry, whose cluster type is their majority vote.
    Entries are ordered pass-2 groups, then pass-1 groups, then unconsolidated centroids.
    """
    raw_accepted_centroids = np.asarray(raw_accepted_centroids)
    n_raw = raw_accepted_centroids.shape[0]

    tier1_groups: dict[int, list[int]] = {}
    tier2_groups: dict[int, list[int]] = {}
    tier3_raw: list[int] = []

    for k in range(n_raw):
        pass1_group = int(tree.raw_to_pass1[k])
        pass2_group = int(tree.pass1_to_pass2[pass1_group])
        if pass2_validity.get(pass2_group, False):
            tier1_groups.setdefault(pass2_group, []).append(k)
        elif pass1_validity.get(pass1_group, False):
            tier2_groups.setdefault(pass1_group, []).append(k)
        else:
            tier3_raw.append(k)

    centroids: list[np.ndarray] = []
    cluster_types: list[str] = []
    source_raw_indices: list[list[int]] = []

    for pass2_group in sorted(tier1_groups):
        raw_indices = sorted(tier1_groups[pass2_group])
        centroids.append(pass2.group_centroids[pass2_group])
        cluster_types.append(_majority_vote([raw_cluster_types[k] for k in raw_indices]))
        source_raw_indices.append(raw_indices)

    for pass1_group in sorted(tier2_groups):
        raw_indices = sorted(tier2_groups[pass1_group])
        centroids.append(pass1.group_centroids[pass1_group])
        cluster_types.append(_majority_vote([raw_cluster_types[k] for k in raw_indices]))
        source_raw_indices.append(raw_indices)

    for k in sorted(tier3_raw):
        centroids.append(raw_accepted_centroids[k])
        cluster_types.append(raw_cluster_types[k])
        source_raw_indices.append([k])

    raw_to_prototype = np.empty(n_raw, dtype=np.int64)
    for prototype, raw_indices in enumerate(source_raw_indices):
        raw_to_prototype[raw_indices] = prototype

    return Vocabulary(
        centroids=np.stack(centroids),
        cluster_types=cluster_types,
        source_raw_indices=source_raw_indices,
        raw_centroids=raw_accepted_centroids,
        raw_to_prototype=raw_to_prototype,
    )
