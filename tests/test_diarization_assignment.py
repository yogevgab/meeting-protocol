import pytest

from meeting_protocol.diarization.assignment import assign_speakers
from meeting_protocol.diarization.base import SpeakerInterval
from meeting_protocol.models import Segment


def _seg(seg_id: int, start: float, end: float) -> Segment:
    return Segment(id=seg_id, start=start, end=end, text=f"s{seg_id}")


def _iv(start: float, end: float, label: str) -> SpeakerInterval:
    return SpeakerInterval(start=start, end=end, label=label)


# ── single-segment dominant-overlap ───────────────────────────────────────────


def test_full_overlap_assigns_label() -> None:
    segs = [_seg(1, 0.0, 5.0)]
    ivs = [_iv(0.0, 5.0, "SPEAKER_00")]

    new_segs, _ = assign_speakers(segs, ivs)

    assert new_segs[0].speaker_id == "SPEAKER_00"


def test_no_overlap_leaves_speaker_id_none() -> None:
    segs = [_seg(1, 0.0, 5.0)]
    ivs = [_iv(10.0, 15.0, "SPEAKER_00")]

    new_segs, speakers = assign_speakers(segs, ivs)

    assert new_segs[0].speaker_id is None
    assert speakers == []


def test_dominant_label_wins_when_segment_spans_two_speakers() -> None:
    # Segment 0–10. Interval A covers 0–7 (7s). Interval B covers 7–10 (3s).
    segs = [_seg(1, 0.0, 10.0)]
    ivs = [_iv(0.0, 7.0, "A"), _iv(7.0, 10.0, "B")]

    new_segs, _ = assign_speakers(segs, ivs)

    assert new_segs[0].speaker_id == "A"


def test_summed_overlap_per_label_handles_split_turns() -> None:
    # Same speaker speaks twice (2s + 2s = 4s of A) inside a segment, with B
    # holding 3s contiguously. A wins by total overlap.
    segs = [_seg(1, 0.0, 10.0)]
    ivs = [
        _iv(0.0, 2.0, "A"),
        _iv(2.0, 5.0, "B"),
        _iv(5.0, 7.0, "A"),
        # 7..10 unattributed
    ]

    new_segs, _ = assign_speakers(segs, ivs)

    assert new_segs[0].speaker_id == "A"


# ── speaker_map name translation ─────────────────────────────────────────────


def test_speaker_map_renames_label_in_speaker_list() -> None:
    segs = [_seg(1, 0.0, 5.0)]
    ivs = [_iv(0.0, 5.0, "SPEAKER_00")]

    new_segs, speakers = assign_speakers(
        segs, ivs, speaker_map={"SPEAKER_00": "Yogev"}
    )

    assert new_segs[0].speaker_id == "SPEAKER_00"  # id stays anonymous-as-key
    assert [(s.id, s.name) for s in speakers] == [("SPEAKER_00", "Yogev")]


def test_unmapped_label_keeps_anonymous_name() -> None:
    segs = [_seg(1, 0.0, 5.0)]
    ivs = [_iv(0.0, 5.0, "SPEAKER_42")]

    _, speakers = assign_speakers(segs, ivs, speaker_map={"SPEAKER_00": "Yogev"})

    assert [(s.id, s.name) for s in speakers] == [("SPEAKER_42", "SPEAKER_42")]


# ── speaker list construction ─────────────────────────────────────────────────


def test_speaker_list_dedupes_repeated_labels() -> None:
    segs = [_seg(1, 0.0, 1.0), _seg(2, 1.0, 2.0), _seg(3, 2.0, 3.0)]
    ivs = [_iv(0.0, 3.0, "SPEAKER_00")]

    _, speakers = assign_speakers(segs, ivs)

    assert len(speakers) == 1
    assert speakers[0].id == "SPEAKER_00"


def test_speaker_list_excludes_unused_labels() -> None:
    """A diarizer interval that doesn't dominate any segment is not in speakers."""
    segs = [_seg(1, 0.0, 5.0)]
    ivs = [
        _iv(0.0, 5.0, "A"),
        _iv(100.0, 105.0, "B"),  # outside any segment
    ]

    _, speakers = assign_speakers(segs, ivs)

    assert [s.id for s in speakers] == ["A"]


def test_multiple_segments_each_get_dominant_speaker() -> None:
    segs = [_seg(1, 0.0, 5.0), _seg(2, 5.0, 10.0), _seg(3, 10.0, 15.0)]
    ivs = [
        _iv(0.0, 5.0, "A"),
        _iv(5.0, 10.0, "B"),
        _iv(10.0, 15.0, "A"),
    ]

    new_segs, speakers = assign_speakers(segs, ivs)

    assert [s.speaker_id for s in new_segs] == ["A", "B", "A"]
    assert {s.id for s in speakers} == {"A", "B"}


# ── empty inputs ──────────────────────────────────────────────────────────────


def test_empty_intervals_returns_segments_unchanged_and_no_speakers() -> None:
    segs = [_seg(1, 0.0, 1.0)]

    new_segs, speakers = assign_speakers(segs, [])

    assert new_segs == segs
    assert speakers == []


def test_empty_segments_returns_empty() -> None:
    new_segs, speakers = assign_speakers([], [_iv(0.0, 5.0, "A")])

    assert new_segs == []
    assert speakers == []


# ── identity / immutability ──────────────────────────────────────────────────


def test_assignment_does_not_mutate_input_segments() -> None:
    seg = _seg(1, 0.0, 5.0)
    segs_in = [seg]
    ivs = [_iv(0.0, 5.0, "A")]

    assign_speakers(segs_in, ivs)

    assert segs_in[0].speaker_id is None  # input unchanged
    assert seg.speaker_id is None


# ── determinism / edge cases ──────────────────────────────────────────────────


def test_tie_break_is_deterministic_by_label_name() -> None:
    """Equal-overlap labels must resolve to the lexicographically-smallest one,
    independent of the order intervals were yielded by the diarizer."""
    segs = [_seg(1, 0.0, 10.0)]
    ivs_order_1 = [_iv(0.0, 5.0, "Z"), _iv(5.0, 10.0, "A")]
    ivs_order_2 = [_iv(5.0, 10.0, "A"), _iv(0.0, 5.0, "Z")]

    new1, _ = assign_speakers(segs, ivs_order_1)
    new2, _ = assign_speakers(segs, ivs_order_2)

    assert new1[0].speaker_id == "A"
    assert new2[0].speaker_id == "A"


def test_zero_duration_segment_gets_no_speaker_id() -> None:
    """A segment with start == end has zero overlap with any interval."""
    segs = [Segment(id=1, start=5.0, end=5.0, text="x")]
    ivs = [_iv(0.0, 10.0, "A")]

    new, speakers = assign_speakers(segs, ivs)

    assert new[0].speaker_id is None
    assert speakers == []


def test_speaker_interval_is_frozen() -> None:
    """SpeakerInterval is a frozen dataclass — mutation must raise."""
    from dataclasses import FrozenInstanceError

    iv = SpeakerInterval(start=0.0, end=1.0, label="A")
    with pytest.raises(FrozenInstanceError):
        iv.label = "B"  # type: ignore[misc]
