"""Segment-level symbolization: central-half majority vote and short-segment merging.

`cpts` follows the convention `[0, t1, ..., T]`: int64, both endpoints included,
strictly increasing. Segment j covers windows [cpts[j], cpts[j+1]).
"""

from __future__ import annotations

import numpy as np

from recursive_prototyping.config.schema import SymbolizationConfig


def _validate_cpts(cpts: np.ndarray, n_windows: int | None = None) -> np.ndarray:
    """Return cpts as int64, or raise ValueError unless it is an integer 1-D array of at
    least 2 entries starting at 0, strictly increasing and ending at `n_windows` if given."""
    cpts = np.asarray(cpts)
    if not np.issubdtype(cpts.dtype, np.integer):
        raise ValueError(f"cpts must be integer-valued, got dtype {cpts.dtype}")
    if cpts.ndim != 1 or len(cpts) < 2:
        raise ValueError(
            f"cpts must be 1-D with at least 2 entries [0, ..., T], got shape {cpts.shape}"
        )
    if cpts[0] != 0:
        raise ValueError(f"cpts must start at 0, got {cpts[0]}")
    if n_windows is not None and cpts[-1] != n_windows:
        raise ValueError(
            f"cpts must end at len(labels) = {n_windows}, got {cpts[-1]}"
        )
    if np.any(np.diff(cpts) <= 0):
        raise ValueError("cpts must be strictly increasing")
    return cpts.astype(np.int64)


def _validate_segment_symbols(
    segment_symbols: np.ndarray, cpts: np.ndarray
) -> np.ndarray:
    """Check one symbol per segment and return the symbols as int64."""
    segment_symbols = np.asarray(segment_symbols)
    if segment_symbols.ndim != 1 or len(segment_symbols) != len(cpts) - 1:
        raise ValueError(
            f"segment_symbols must be 1-D with len(cpts) - 1 = {len(cpts) - 1} "
            f"entries, got shape {segment_symbols.shape}"
        )
    return segment_symbols.astype(np.int64)


def central_half_vote(labels: np.ndarray, cpts: np.ndarray) -> np.ndarray:
    """Majority-vote one symbol per segment over the segment's central half.

    The vote runs over [start + trim, end - trim) with trim = length // 4 (floor), so
    segments of length 1-3 vote over all their windows. Background (-1) votes like any
    symbol; ties go to the smallest symbol value, so -1 wins a tie. Returns int64
    [len(cpts) - 1].
    """
    labels = np.asarray(labels)
    cpts = _validate_cpts(cpts, n_windows=len(labels))

    n_segments = len(cpts) - 1
    segment_symbols = np.empty(n_segments, dtype=np.int64)
    for j in range(n_segments):
        start, end = cpts[j], cpts[j + 1]
        trim = (end - start) // 4
        values, counts = np.unique(labels[start + trim : end - trim], return_counts=True)
        segment_symbols[j] = values[np.argmax(counts)]
    return segment_symbols


def merge_short_segments(
    cpts: np.ndarray,
    segment_symbols: np.ndarray,
    min_segment_length: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge short segments flanked by identical symbols, in one left-to-right pass.

    An interior segment shorter than `min_segment_length` whose two neighbors carry the
    same symbol is merged with them under that symbol. The merged segment is the left
    neighbor of the next candidate, so one merge can enable the next; the pass is not
    repeated. Returns (new_cpts, new_symbols), both int64.
    """
    cpts = _validate_cpts(cpts)
    segment_symbols = _validate_segment_symbols(segment_symbols, cpts)

    working = [
        [int(cpts[j]), int(cpts[j + 1]), int(segment_symbols[j])]
        for j in range(len(segment_symbols))
    ]
    i = 1
    while i < len(working) - 1:
        start, end, _ = working[i]
        left_start, _, left_symbol = working[i - 1]
        _, right_end, right_symbol = working[i + 1]
        if end - start < min_segment_length and left_symbol == right_symbol:
            working[i - 1 : i + 2] = [[left_start, right_end, left_symbol]]
            # The merged segment is now at i - 1, so index i is the next candidate.
        else:
            i += 1

    new_cpts = np.array([working[0][0]] + [seg[1] for seg in working], dtype=np.int64)
    new_symbols = np.array([seg[2] for seg in working], dtype=np.int64)
    return new_cpts, new_symbols


def symbolize(
    labels: np.ndarray, cpts: np.ndarray, config: SymbolizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Central-half vote then short-segment merging; returns (final_cpts, final_symbols)."""
    segment_symbols = central_half_vote(labels, cpts)
    return merge_short_segments(
        cpts, segment_symbols, min_segment_length=config.min_segment_length
    )


def expand_to_windows(cpts: np.ndarray, segment_symbols: np.ndarray) -> np.ndarray:
    """Repeat each segment's symbol over its windows; returns int64 [cpts[-1]]."""
    cpts = _validate_cpts(cpts)
    segment_symbols = _validate_segment_symbols(segment_symbols, cpts)
    return np.repeat(segment_symbols, np.diff(cpts))
