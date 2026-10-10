"""Configuration dataclasses for the three pipeline steps.

Defaults are the values used in the paper (Perochon & Oudre, ICPR 2026).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrototypingConfig:
    """Prototyping: recursive loop and vocabulary consolidation."""

    embedding_dim: int = 1408
    """VideoMAE-v2 embedding dimension."""

    n_clusters_per_round: int = 100
    """C: k-means clusters per round."""

    n_rounds: int = 8
    """P: number of prototyping rounds."""

    max_accepted_prototypes: int | None = None
    """K_max: stop once MORE than this many prototypes are accepted (checked before each
    round, as the paper's SM states "exceeds K_max"); None runs all rounds."""

    residual_distance_threshold: float = 0.3
    """d_min: samples of an accepted cluster further than this cosine distance from its
    centroid return to the residual set (samples of ambiguous clusters always do)."""

    faiss_niter: int = 250
    """Faiss k-means iterations."""

    faiss_nredo: int = 2
    """Faiss k-means restarts (the best is kept)."""

    faiss_seed: int = 47
    """Faiss RNG seed."""

    faiss_max_points_per_centroid: int = 20000
    """Faiss k-means subsampling cap per centroid."""

    n_probe_neighbors: int = 30
    """Samples shown per side (nearest, furthest) per cluster in a probe."""

    n_probe_participants_target: int = 30
    """Target number of distinct participants per probe."""

    probe_image_ncols: int = 10
    """Columns of the rendered probe image (display only)."""

    annotation_categories: tuple[str, ...] = (
        "task-definitive",
        "exo-definitive",
        "task-ambiguous",
        "exo-ambiguous",
        "noise",
    )
    """Four labels plus 'noise', which is discarded."""

    consolidation_linkage: str = "complete"
    """HAC linkage for vocabulary consolidation."""

    consolidation_metric: str = "cosine"
    """HAC distance metric; only "cosine" is implemented (others raise NotImplementedError)."""

    random_seed: int = 47
    """Seed for non-Faiss randomness (e.g. probe sampling)."""


@dataclass
class SegmentationConfig:
    """Segmentation: per-participant kernel change-point detection + slope-heuristic penalty."""

    kernel: str = "rbf"
    """Kernel for ruptures.KernelCPD: "rbf" or "linear"."""

    gamma_temperature: float = 1.0
    """RBF temperature: gamma = 1 / (2 * temperature * sigma^2), with sigma the median
    pairwise distance of the participant's sequence. Raising it lowers gamma."""

    min_segment_size: int = 2
    """Minimum segment length in embeddings (ruptures min_size)."""

    jump: int = 1
    """ruptures jump. KernelCPD always uses 1, so other values have no effect."""

    m_max: int | None = None
    """Top of the calibration grid; None derives it from the video duration."""

    seconds_per_segment_default: float = 2.0
    """Seconds per segment used to derive m_max = duration_seconds / this value when m_max
    is unset. Not the embedding hop (the ``hop_seconds`` argument, 0.32 s)."""

    m_grid_size: int = 30
    """Number of change-point counts in the calibration grid."""

    m_grid_low_frac: float = 0.6
    """Lower end of the calibration grid as a fraction of m_max."""

    random_seed: int = 47
    """Unused: segmentation has no stochastic step."""


@dataclass
class SymbolizationConfig:
    """Symbolization: nearest-centroid symbolic assignment + segment-level majority vote."""

    background_n_sigma: float = 1.0
    """Windows whose assignment distance exceeds mean + n_sigma * std become -1."""

    min_segment_length: int = 4
    """Segments shorter than this (in embeddings) between two identical symbols are merged."""

    random_seed: int = 47
    """Unused: symbolization has no stochastic step."""
