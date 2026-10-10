"""Placeholder frame renderer for the Visual Probing Mechanism on synthetic data.

`SyntheticFrameSource` satisfies the `FrameSource` protocol of `prototyping/vpm.py`
without any video dependency.
"""

from __future__ import annotations

import hashlib

import numpy as np

from recursive_prototyping.synthetic.generator import ParticipantSynth

# Colors for action clusters, cycled when there are more clusters than entries.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),  # red
    (60, 180, 75),  # green
    (255, 225, 25),  # yellow
    (0, 130, 200),  # blue
    (245, 130, 48),  # orange
    (145, 30, 180),  # purple
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (210, 245, 60),  # lime
    (0, 128, 128),  # teal
)

_BACKGROUND_BASE = (128, 128, 128)  # gray base of background windows
_BACKGROUND_SPECKLE_AMPLITUDE = 40  # +/- range of the speckle noise
_MARKER_FRACTION = 0.15  # marker square side as a fraction of image_size


def _deterministic_seed(*parts: object) -> int:
    """Integer seed derived from `parts`, stable across processes (unlike `hash()`)."""
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big")


class SyntheticFrameSource:
    """Renders a small image per (participant_id, embedding_index) from the oracle cluster ids."""

    def __init__(self, participants: dict[str, ParticipantSynth], image_size: int = 64):
        """`participants` is the dict returned by `generate_synthetic_embeddings`."""
        self._participants = participants
        self._image_size = image_size

    def central_frame(self, participant_id: str, embedding_index: int) -> np.ndarray:
        """Render the frame for one (participant, window) pair.

        Action clusters are a solid palette color; background windows are gray with
        speckle seeded from (participant_id, embedding_index). A corner marker holds the
        cluster id mod 256, so clusters sharing a palette color stay distinguishable.
        """
        oracle_cluster_ids = self._participants[participant_id].oracle_cluster_ids
        cluster_id = int(oracle_cluster_ids[embedding_index])

        if cluster_id == -1:
            frame = self._render_background(participant_id, embedding_index)
        else:
            frame = self._render_cluster(cluster_id)

        self._burn_marker(frame, cluster_id)
        return frame

    def _render_cluster(self, cluster_id: int) -> np.ndarray:
        color = _PALETTE[cluster_id % len(_PALETTE)]
        return np.full((self._image_size, self._image_size, 3), color, dtype=np.uint8)

    def _render_background(self, participant_id: str, embedding_index: int) -> np.ndarray:
        seed = _deterministic_seed(participant_id, embedding_index)
        rng = np.random.default_rng(seed)

        shape = (self._image_size, self._image_size, 3)
        speckle = rng.integers(
            -_BACKGROUND_SPECKLE_AMPLITUDE, _BACKGROUND_SPECKLE_AMPLITUDE + 1, size=shape
        )
        noisy = np.array(_BACKGROUND_BASE, dtype=np.int16) + speckle
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _burn_marker(self, frame: np.ndarray, cluster_id: int) -> None:
        marker_side = max(1, int(round(self._image_size * _MARKER_FRACTION)))
        marker_value = cluster_id % 256  # background (-1) becomes 255
        frame[:marker_side, :marker_side, :] = marker_value
