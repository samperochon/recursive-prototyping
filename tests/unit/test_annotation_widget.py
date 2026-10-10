"""Unit tests for recursive_prototyping.prototyping.annotation_widget."""

import sys
import types
from pathlib import Path

import pytest

from recursive_prototyping.prototyping.annotation_widget import (
    annotate_probe_images,
    collect_annotations,
    fake_annotate,
)


def test_fake_annotate_matches_rule_and_preserves_order():
    image_paths = [Path(f"probe_{i}.png") for i in [2, 0, 1]]

    def rule(path: Path) -> str:
        return "task-definitive" if "0" in path.stem else "exo-definitive"

    result = fake_annotate(image_paths, rule)

    assert [path for path, _ in result] == image_paths
    assert [label for _, label in result] == [rule(path) for path in image_paths]
    assert result == [
        (Path("probe_2.png"), "exo-definitive"),
        (Path("probe_0.png"), "task-definitive"),
        (Path("probe_1.png"), "exo-definitive"),
    ]


def test_fake_annotate_empty_input():
    assert fake_annotate([], lambda path: "noise") == []


def test_annotate_probe_images_raises_import_error_with_monkeypatched_missing_pigeon(
    monkeypatch,
):
    # Robustness check: force `import pigeon` to fail even if pigeon-jupyter happens to be
    # installed in the environment running this test.
    monkeypatch.setitem(sys.modules, "pigeon", None)

    with pytest.raises(ImportError, match=r"recursive-prototyping\[annotation\]"):
        annotate_probe_images([Path("probe_0.png")], ("task-definitive", "noise"))


# --------------------------------------------------------------------------
# The pigeon contract, pinned against the pigeon-jupyter 0.1.0 API
# (see annotation_widget's module docstring).


@pytest.fixture
def stub_ipython(monkeypatch):
    """annotate_probe_images imports IPython.display, which only the `annotation` and
    `notebook` extras install; a `[dev]`-only environment must still run these tests."""
    display = types.ModuleType("IPython.display")
    display.Image = lambda filename: filename
    display.display = lambda obj: None
    ipython = types.ModuleType("IPython")
    ipython.display = display
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)


class _FakePigeon:
    """Stands in for pigeon.annotate, reproducing the two behaviours that matter:
    the returned list is EMPTY at return time, and it is the same object the
    widget callbacks later append to."""

    def __init__(self):
        self.calls = []
        self.live = []

    def annotate(self, examples, options=None, include_skip=True, display_fn=None):
        self.calls.append(
            {"examples": list(examples), "options": options, "include_skip": include_skip}
        )
        self.live = []
        return self.live

    def click_through(self, labels):
        for example, label in zip(self.calls[-1]["examples"], labels):
            self.live.append((example, label))


def test_annotate_probe_images_returns_the_live_empty_list_and_disables_skip(monkeypatch, stub_ipython):
    pigeon = _FakePigeon()
    monkeypatch.setitem(sys.modules, "pigeon", pigeon)
    paths = [Path("probe_0.png"), Path("probe_1.png")]

    live = annotate_probe_images(paths, ("task-definitive", "noise"))

    # Empty at return time: pigeon fills the list only as the images are clicked.
    assert live == []
    call = pigeon.calls[-1]
    assert call["examples"] == ["probe_0.png", "probe_1.png"]
    assert call["options"] == ["task-definitive", "noise"]
    # skip must be off: a skipped image yields no tuple and would KeyError in
    # run_recursive_loop's path-keyed lookup.
    assert call["include_skip"] is False

    # ...and it fills in afterwards, in place.
    pigeon.click_through(["task-definitive", "noise"])
    assert live == [("probe_0.png", "task-definitive"), ("probe_1.png", "noise")]


def test_collect_annotations_returns_image_paths_order(monkeypatch, stub_ipython):
    pigeon = _FakePigeon()
    monkeypatch.setitem(sys.modules, "pigeon", pigeon)
    paths = [Path("probe_0.png"), Path("probe_1.png")]

    live = annotate_probe_images(paths, ("task-definitive", "noise"))
    # Annotated back-to-front: collect_annotations must not inherit pigeon's order.
    live.extend([("probe_1.png", "noise"), ("probe_0.png", "task-definitive")])

    assert collect_annotations(live, paths) == [
        (Path("probe_0.png"), "task-definitive"),
        (Path("probe_1.png"), "noise"),
    ]


def test_collect_annotations_rejects_an_unfinished_pass():
    paths = [Path("probe_0.png"), Path("probe_1.png")]

    # The expected user error: running the collect cell before clicking every image.
    with pytest.raises(ValueError, match="not annotated yet"):
        collect_annotations([("probe_0.png", "noise")], paths)

    with pytest.raises(ValueError, match="not annotated yet"):
        collect_annotations([], paths)


def test_collect_annotations_rejects_a_foreign_path():
    with pytest.raises(ValueError, match="not in image_paths"):
        collect_annotations(
            [("probe_9.png", "noise")], [Path("probe_0.png")]
        )
