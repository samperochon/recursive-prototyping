"""Human validity verdicts for consolidated groups (meta-prototypes).

The paper rejects a group when a rater judges that any pair of its constituent centroids
shows distinct actions. ``render_meta_probe_images`` stacks each group's member probe images
into one composite image for that inspection, singleton groups included, and
``collect_validity`` turns the verdicts into the validity dict
``assemble_final_vocabulary`` takes, in place of the ``consolidation.compute_group_validity``
proxy.

A rejected group is only left unmerged. A further verdict that sends such a group back for
re-clustering, to split it into coherent sub-groups, is not implemented.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: Verdicts for a consolidated group; distinct from the cluster categories of the recursive loop.
META_CATEGORIES: tuple[str, ...] = ("valid", "ambiguous")

#: The only verdict that counts as valid.
_VALID_VERDICT = "valid"

_GROUP_RE = re.compile(r"_group_(\d+)\.png$")


def render_meta_probe_images(
    assignment: np.ndarray,
    member_probe_paths: list[Path],
    out_dir: str | Path,
    prefix: str,
    max_height: int = 1800,
) -> tuple[list[Path], list[list[int]]]:
    """Render one composite probe image per consolidated group.

    Item ``i`` of the pass belongs to group ``assignment[i]``, and ``member_probe_paths[i]``
    is its rendered probe image: the cluster probe images for pass 1, the pass-1 composites
    from an earlier call for pass 2. Members are stacked vertically at a common width; a composite
    taller than ``max_height`` is scaled down, never cropped.

    Returns ``(probe_paths, members_by_group)``, both indexed by group id.

    Raises:
        ValueError: if ``assignment`` and ``member_probe_paths`` disagree in length.
    """
    from PIL import Image

    assignment = np.asarray(assignment)
    if len(assignment) != len(member_probe_paths):
        raise ValueError(
            f"assignment has {len(assignment)} items but member_probe_paths has "
            f"{len(member_probe_paths)} -- they must be the same pass's items, in the same order."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_groups = int(assignment.max()) + 1 if len(assignment) else 0
    probe_paths: list[Path] = []
    members_by_group: list[list[int]] = []

    for g in range(n_groups):
        members = [int(i) for i in np.flatnonzero(assignment == g)]
        members_by_group.append(members)

        tiles = [Image.open(member_probe_paths[i]).convert("RGB") for i in members]
        width = max(tile.width for tile in tiles)
        scaled = [
            tile if tile.width == width
            else tile.resize((width, round(tile.height * width / tile.width)), Image.LANCZOS)
            for tile in tiles
        ]

        rule = 4
        height = sum(tile.height for tile in scaled) + rule * (len(scaled) - 1)
        composite = Image.new("RGB", (width, height), "white")
        y = 0
        for tile in scaled:
            composite.paste(tile, (0, y))
            y += tile.height + rule

        if composite.height > max_height:
            scale = max_height / composite.height
            composite = composite.resize(
                (max(1, round(composite.width * scale)), max_height), Image.LANCZOS
            )

        out_path = out_dir / f"{prefix}_group_{g}.png"
        composite.save(out_path)
        for tile in tiles:
            tile.close()
        probe_paths.append(out_path)

    return probe_paths, members_by_group


def collect_validity(
    annotations: list[tuple[Path, str]], probe_paths: list[Path]
) -> dict[int, bool]:
    """Turn a completed meta-prototype pass into ``assemble_final_vocabulary``'s validity dict.

    ``annotations`` is what ``annotation_widget.collect_annotations`` returns. Group ids are
    parsed from the filenames ``render_meta_probe_images`` wrote. Only ``"valid"`` maps to True.

    Raises:
        ValueError: on an unknown verdict, an unparseable path, or a group left without one.
    """
    verdict_by_path = {Path(path): verdict for path, verdict in annotations}

    validity: dict[int, bool] = {}
    for path in probe_paths:
        match = _GROUP_RE.search(Path(path).name)
        if match is None:
            raise ValueError(
                f"{path} is not a meta-probe path -- expected a name ending '_group_<id>.png' "
                "as written by render_meta_probe_images."
            )
        if Path(path) not in verdict_by_path:
            raise ValueError(f"group {match.group(1)} ({path}) has no verdict.")

        verdict = verdict_by_path[Path(path)]
        if verdict not in META_CATEGORIES:
            raise ValueError(
                f"{verdict!r} is not one of {META_CATEGORIES} -- was this pass run with the "
                "five cluster categories by mistake?"
            )
        validity[int(match.group(1))] = verdict == _VALID_VERDICT

    return validity
