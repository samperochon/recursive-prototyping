"""Annotation entry points: turn probe image paths into (image_path, category) labels.

- ``annotate_probe_images`` + ``collect_annotations``: interactive, using the optional
  ``pigeon-jupyter`` widget (the ``annotation`` extra), imported lazily so this module
  works without it.
- ``fake_annotate``: a scripted annotator for tests and headless runs.

``pigeon.annotate`` returns an empty list immediately and fills it as the user clicks, so a
live pass needs two notebook cells: ``annotate_probe_images`` in one, ``collect_annotations``
in a later one. ``annotate_probe_images`` is therefore not an ``AnnotateFn`` and cannot be
passed to ``run_recursive_loop``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def annotate_probe_images(image_paths: list[Path], categories: tuple[str, ...]) -> list:
    """Open the ``pigeon-jupyter`` widget over ``image_paths``, one button per category.

    Returns the list pigeon appends ``(path_str, label)`` pairs to: it is empty on return
    and fills as the user clicks. Pass it to ``collect_annotations`` in a later cell.

    The skip button is disabled because every image needs a verdict; use ``noise`` for a
    cluster that deserves none.

    Raises:
        ImportError: if ``pigeon-jupyter`` is not installed.
    """
    try:
        import pigeon
    except ImportError as exc:
        raise ImportError(
            "pigeon-jupyter is required for interactive annotation. "
            "Install it with: pip install recursive-prototyping[annotation]"
        ) from exc

    from IPython.display import Image, display

    return pigeon.annotate(
        [str(path) for path in image_paths],
        options=list(categories),
        include_skip=False,
        display_fn=lambda filename: display(Image(filename)),
    )


def collect_annotations(
    annotations: list, image_paths: list[Path]
) -> list[tuple[Path, str]]:
    """Convert a completed ``annotate_probe_images`` list into ``[(Path, category)]`` pairs,
    in ``image_paths`` order.

    Raises:
        ValueError: if a path in ``image_paths`` has no annotation yet, or if
            ``annotations`` has a path that is not in ``image_paths``.
    """
    label_by_path = {Path(path): label for path, label in annotations}

    unexpected = set(label_by_path) - set(image_paths)
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} annotated path(s) are not in image_paths, e.g. "
            f"{sorted(str(p) for p in unexpected)[:3]}"
        )

    missing = [path for path in image_paths if path not in label_by_path]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(image_paths)} probe images are not annotated yet "
            f"(e.g. {[str(p) for p in missing[:3]]}). Work through every image in the "
            "widget before calling collect_annotations."
        )

    return [(path, label_by_path[path]) for path in image_paths]


def fake_annotate(
    image_paths: list[Path], rule: Callable[[Path], str]
) -> list[tuple[Path, str]]:
    """Scripted annotator: applies ``rule`` to each path, in order."""
    return [(path, rule(path)) for path in image_paths]
