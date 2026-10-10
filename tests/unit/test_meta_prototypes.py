"""Unit tests for recursive_prototyping.prototyping.meta_prototypes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recursive_prototyping.prototyping.consolidation import assemble_final_vocabulary
from recursive_prototyping.prototyping.meta_prototypes import (
    META_CATEGORIES,
    collect_validity,
    render_meta_probe_images,
)


@pytest.fixture
def member_images(tmp_path) -> list[Path]:
    """Six distinct solid-colour stand-ins for cluster probe images, of differing sizes."""
    colours = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (0, 200, 200), (200, 0, 200)]
    paths = []
    for i, colour in enumerate(colours):
        path = tmp_path / f"member_{i}.png"
        Image.new("RGB", (100 + 10 * i, 120), colour).save(path)
        paths.append(path)
    return paths


def test_every_group_gets_an_image_containing_all_and_only_its_members(member_images, tmp_path):
    """The verdict asks whether ANY PAIR of constituent centroids conflicts, so the composite
    has to show every constituent -- a sampled or truncated composite would let a rater pass a
    group whose conflicting pair was never rendered.
    """
    assignment = np.array([0, 0, 1, 1, 1, 2])
    paths, members = render_meta_probe_images(assignment, member_images, tmp_path / "meta", "pass1")

    assert len(paths) == 3
    assert members == [[0, 1], [2, 3, 4], [5]]
    assert all(path.exists() for path in paths)

    # Group 1 holds three members, so its composite must be taller than group 2's single one.
    with Image.open(paths[1]) as many, Image.open(paths[2]) as one:
        assert many.height > one.height

    # Every member's colour must actually appear in its group's composite.
    for group_id, group_members in enumerate(members):
        with Image.open(paths[group_id]) as composite:
            present = {tuple(c) for c in np.unique(np.asarray(composite).reshape(-1, 3), axis=0)}
        for member in group_members:
            with Image.open(member_images[member]) as tile:
                colour = tuple(np.asarray(tile)[0, 0])
            assert any(
                sum(abs(int(a) - int(b)) for a, b in zip(colour, seen)) < 60 for seen in present
            ), f"member {member}'s frames are missing from group {group_id}'s composite"


def test_singletons_get_an_image_rather_than_being_auto_passed(member_images, tmp_path):
    """The paper makes a singleton trivially valid, the headless proxy marks it invalid
    (docs/METHOD.md, section 4). With a human in the loop neither guess is needed -- so a singleton
    must be RENDERED and asked about, not decided for the rater in either direction.
    """
    assignment = np.array([0, 1, 2, 3, 4, 5])
    paths, members = render_meta_probe_images(assignment, member_images, tmp_path / "meta", "pass1")

    assert members == [[i] for i in range(6)]
    assert len(paths) == 6, "every singleton group must still be put in front of the rater"


def test_composites_are_capped_in_height_without_dropping_members(member_images, tmp_path):
    assignment = np.zeros(6, dtype=int)
    paths, members = render_meta_probe_images(
        assignment, member_images, tmp_path / "meta", "pass1", max_height=200
    )

    assert members == [[0, 1, 2, 3, 4, 5]], "capping height must not drop constituents"
    with Image.open(paths[0]) as composite:
        assert composite.height == 200


def test_pass2_composites_compose_over_pass1_composites(member_images, tmp_path):
    """Pass 2's items are pass-1 GROUPS. Feeding it the pass-1 composites (not raw member
    images) is what makes a pass-2 verdict a judgment about every raw centroid beneath it,
    not only about its direct items.
    """
    pass1_paths, _ = render_meta_probe_images(
        np.array([0, 0, 1, 1, 2, 2]), member_images, tmp_path / "meta", "pass1"
    )
    pass2_paths, pass2_members = render_meta_probe_images(
        np.array([0, 0, 1]), pass1_paths, tmp_path / "meta", "pass2"
    )

    assert pass2_members == [[0, 1], [2]]
    # Pass-2 group 0 covers pass-1 groups 0 and 1, i.e. raw members 0-3: all four colours.
    with Image.open(pass2_paths[0]) as composite:
        present = np.unique(np.asarray(composite).reshape(-1, 3), axis=0)
    for colour in [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]:
        assert any(
            sum(abs(int(a) - int(b)) for a, b in zip(colour, seen)) < 60 for seen in present
        ), f"{colour} (a raw member two levels down) is missing from the pass-2 composite"


def test_collect_validity_maps_only_valid_to_true(member_images, tmp_path):
    paths, _ = render_meta_probe_images(
        np.array([0, 1, 2]), member_images[:3], tmp_path / "meta", "pass1"
    )
    annotations = list(zip(paths, ["valid", "ambiguous", "ambiguous"]))

    assert collect_validity(annotations, paths) == {0: True, 1: False, 2: False}


def test_collect_validity_rejects_the_cluster_category_set(member_images, tmp_path):
    """Running the meta pass with the five cluster categories is a plausible copy-paste slip, and
    it would otherwise silently mark every group invalid.
    """
    paths, _ = render_meta_probe_images(
        np.array([0, 1]), member_images[:2], tmp_path / "meta", "pass1"
    )
    annotations = list(zip(paths, ["task-definitive", "noise"]))

    with pytest.raises(ValueError, match="not one of"):
        collect_validity(annotations, paths)


def test_collect_validity_refuses_a_group_left_unjudged(member_images, tmp_path):
    paths, _ = render_meta_probe_images(
        np.array([0, 1]), member_images[:2], tmp_path / "meta", "pass1"
    )

    with pytest.raises(ValueError, match="no verdict"):
        collect_validity([(paths[0], "valid")], paths)


def test_human_validity_dicts_drop_into_assemble_final_vocabulary_unchanged():
    """The whole point of the module: no signature downstream has to change. A human validity
    dict is shape-compatible with compute_group_validity's, so it substitutes directly.
    """
    from recursive_prototyping.prototyping.consolidation import (
        ConsolidationResult,
        MembershipTree,
    )

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((6, 8))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    pass1 = ConsolidationResult(
        group_centroids=raw[:3], assignment=np.array([0, 0, 1, 1, 2, 2]), silhouette_by_g={3: 0.5},
        best_g=3,
    )
    pass2 = ConsolidationResult(
        group_centroids=raw[:2], assignment=np.array([0, 0, 1]), silhouette_by_g={2: 0.4}, best_g=2,
    )
    tree = MembershipTree(raw_to_pass1=pass1.assignment, pass1_to_pass2=pass2.assignment)

    vocabulary = assemble_final_vocabulary(
        tree=tree,
        pass1_validity={0: True, 1: False, 2: True},   # as a rater would have produced
        pass2_validity={0: True, 1: False},
        pass1=pass1,
        pass2=pass2,
        raw_accepted_centroids=raw,
        raw_cluster_types=["task-definitive"] * 6,
    )

    covered = sorted(i for entry in vocabulary.source_raw_indices for i in entry)
    assert covered == list(range(6)), "a centroid is never lost"
    assert len(META_CATEGORIES) == 2
