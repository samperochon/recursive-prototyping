"""Unit tests for recursive_prototyping.symbolization.segments (pure segment math).

All cases are hand-built with independently-computed expected outputs;
symbolize's upstream inputs (real assignment labels) are covered by the
integration tests, not here.
"""

import numpy as np
import pytest

from recursive_prototyping.config.schema import SymbolizationConfig
from recursive_prototyping.symbolization.segments import (
    central_half_vote,
    expand_to_windows,
    merge_short_segments,
    symbolize,
)


def test_central_half_vote_excludes_outer_quarters():
    # Len-8 segment, trim 2: outer quarters carry 5 (5 windows of it in
    # total), central half carries [2, 2, 2, 5]. A full-segment vote would
    # return 5 (5 vs 3); the central-half vote must return 2.
    labels = np.array([5, 5, 2, 2, 2, 5, 5, 5])
    np.testing.assert_array_equal(central_half_vote(labels, [0, 8]), [2])


def test_central_half_vote_trim_by_segment_length():
    # trim = len // 4 per segment: len 1-3 -> trim 0 (whole segment),
    # len 4 -> middle 2, len 8 -> middle 4. Hand-computed per segment:
    # [3] -> 3; [5, 3] -> tie -> 3; [8, 5, 8] -> 8 (trim 1 would give 5);
    # [0, 1, 1, 0] -> central [1, 1] -> 1 (full vote would tie -> 0);
    # [0, 0, 1, 1, 1, 1, 0, 0] -> central [1, 1, 1, 1] -> 1 (full vote
    # would tie -> 0).
    labels = np.array([3, 5, 3, 8, 5, 8, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    cpts = np.array([0, 1, 3, 6, 10, 18])

    segment_symbols = central_half_vote(labels, cpts)

    assert segment_symbols.dtype == np.int64
    np.testing.assert_array_equal(segment_symbols, [3, 3, 8, 1, 1])


def test_central_half_vote_tie_break_smallest_symbol_wins():
    # Exact tie between prototypes 4 and 9 -> smallest (4) wins.
    np.testing.assert_array_equal(central_half_vote(np.array([9, 4]), [0, 2]), [4])


def test_central_half_vote_tie_break_background_beats_prototype():
    # Documented convention: -1 is the smallest symbol, so it wins any tie
    # against a prototype.
    np.testing.assert_array_equal(central_half_vote(np.array([3, -1]), [0, 2]), [-1])


def test_merge_short_segments_merges_short_middle_with_equal_flanks():
    # [A(5), B(2), A(4)]: B is interior, len 2 < 4, flanks both A=2 ->
    # the three segments become one A spanning [0, 11).
    new_cpts, new_symbols = merge_short_segments(
        np.array([0, 5, 7, 11]), np.array([2, 7, 2])
    )

    assert new_cpts.dtype == np.int64
    assert new_symbols.dtype == np.int64
    np.testing.assert_array_equal(new_cpts, [0, 11])
    np.testing.assert_array_equal(new_symbols, [2])


def test_merge_short_segments_different_flanks_untouched():
    # [A(5), B(2), C(4)]: flank symbols differ (2 vs 3) -> no merge.
    new_cpts, new_symbols = merge_short_segments(
        np.array([0, 5, 7, 11]), np.array([2, 7, 3])
    )
    np.testing.assert_array_equal(new_cpts, [0, 5, 7, 11])
    np.testing.assert_array_equal(new_symbols, [2, 7, 3])


def test_merge_short_segments_first_and_last_never_merged():
    # Short FIRST segment [B(2), A(5), B(4)] and short LAST segment
    # [B(4), A(5), B(2)]: boundary segments have no both-side flank.
    new_cpts, new_symbols = merge_short_segments(
        np.array([0, 2, 7, 11]), np.array([7, 2, 7])
    )
    np.testing.assert_array_equal(new_cpts, [0, 2, 7, 11])
    np.testing.assert_array_equal(new_symbols, [7, 2, 7])

    new_cpts, new_symbols = merge_short_segments(
        np.array([0, 4, 9, 11]), np.array([7, 2, 7])
    )
    np.testing.assert_array_equal(new_cpts, [0, 4, 9, 11])
    np.testing.assert_array_equal(new_symbols, [7, 2, 7])


def test_merge_short_segments_adjacent_same_symbol_not_collapsed():
    # Only the short-middle-equal-flanks pattern triggers: two adjacent
    # same-symbol segments (one of them short) stay separate.
    new_cpts, new_symbols = merge_short_segments(np.array([0, 2, 7]), np.array([4, 4]))
    np.testing.assert_array_equal(new_cpts, [0, 2, 7])
    np.testing.assert_array_equal(new_symbols, [4, 4])


def test_merge_short_segments_chained_merges_in_single_pass():
    # [A(5), B(2), A(2), B(3), A(6)] with A=7, B=3. Working-list pass:
    # i=1 merges (0,5,7)+(5,7,3)+(7,9,7) -> (0,9,7); the pass continues
    # from the merged result, so (9,12,3) is now interior with flanks
    # (0,9,7) and (12,18,7) and merges too -> single (0,18,7).
    new_cpts, new_symbols = merge_short_segments(
        np.array([0, 5, 7, 9, 12, 18]), np.array([7, 3, 7, 3, 7])
    )
    np.testing.assert_array_equal(new_cpts, [0, 18])
    np.testing.assert_array_equal(new_symbols, [7])


def test_merge_short_segments_no_fixpoint_iteration():
    # [A(4), B(1), C(1), B(1), A(4)] with A=2, B=7, C=9: i=1 does not
    # merge (flanks 2 vs 9); i=2 merges the three middles into (4,7,7).
    # That merged B(3) is short with equal A flanks, but it sits behind
    # the scan position -- a single pass must NOT revisit it (no
    # repeat-until-fixpoint).
    new_cpts, new_symbols = merge_short_segments(
        np.array([0, 4, 5, 6, 7, 11]), np.array([2, 7, 9, 7, 2])
    )
    np.testing.assert_array_equal(new_cpts, [0, 4, 7, 11])
    np.testing.assert_array_equal(new_symbols, [2, 7, 2])


def test_merge_short_segments_length_equal_to_min_not_merged():
    # Strict <: middle of len 4 with min_segment_length=4 stays; the same
    # input with min_segment_length=5 merges.
    cpts = np.array([0, 4, 8, 12])
    segment_symbols = np.array([2, 7, 2])

    new_cpts, new_symbols = merge_short_segments(cpts, segment_symbols, min_segment_length=4)
    np.testing.assert_array_equal(new_cpts, [0, 4, 8, 12])
    np.testing.assert_array_equal(new_symbols, [2, 7, 2])

    new_cpts, new_symbols = merge_short_segments(cpts, segment_symbols, min_segment_length=5)
    np.testing.assert_array_equal(new_cpts, [0, 12])
    np.testing.assert_array_equal(new_symbols, [2])


def test_symbolize_threads_min_segment_length_from_config():
    # Vote gives [2, 7, 2] with a len-2 middle (hand-computed: len-5
    # segments trim to their all-2 central 3, the len-2 segment votes
    # [7, 7]). Default config (min 4) merges the middle; min 2 does not
    # (strict <: 2 < 2 is false).
    labels = np.array([2] * 5 + [7] * 2 + [2] * 5)
    cpts = np.array([0, 5, 7, 12])

    final_cpts, final_symbols = symbolize(labels, cpts, SymbolizationConfig())
    np.testing.assert_array_equal(final_cpts, [0, 12])
    np.testing.assert_array_equal(final_symbols, [2])

    final_cpts, final_symbols = symbolize(
        labels, cpts, SymbolizationConfig(min_segment_length=2)
    )
    np.testing.assert_array_equal(final_cpts, [0, 5, 7, 12])
    np.testing.assert_array_equal(final_symbols, [2, 7, 2])


def test_expand_to_windows_round_trip():
    cpts = np.array([0, 3, 7, 12])
    segment_symbols = np.array([4, -1, 9])

    windows = expand_to_windows(cpts, segment_symbols)

    assert windows.dtype == np.int64
    assert len(windows) == cpts[-1]
    np.testing.assert_array_equal(windows, [4, 4, 4, -1, -1, -1, -1, 9, 9, 9, 9, 9])
    # Run lengths match np.diff(cpts), and constant-per-segment windows
    # vote back to the original symbols.
    for j, symbol in enumerate(segment_symbols):
        np.testing.assert_array_equal(
            windows[cpts[j] : cpts[j + 1]], [symbol] * (cpts[j + 1] - cpts[j])
        )
    np.testing.assert_array_equal(central_half_vote(windows, cpts), segment_symbols)


def test_cpts_validation_errors():
    labels = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError, match="start at 0"):
        central_half_vote(labels, np.array([1, 5]))
    with pytest.raises(ValueError, match="end at len"):
        central_half_vote(labels, np.array([0, 4]))
    with pytest.raises(ValueError, match="strictly increasing"):
        central_half_vote(labels, np.array([0, 3, 3, 5]))
    with pytest.raises(ValueError, match="integer-valued"):
        central_half_vote(labels, np.array([0.0, 5.0]))


def test_expand_to_windows_symbol_count_mismatch_raises():
    with pytest.raises(ValueError, match="len\\(cpts\\) - 1"):
        expand_to_windows(np.array([0, 3, 7]), np.array([4]))
