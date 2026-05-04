"""Unit tests for merge_transcripts — multi-clip timestamp stitching."""

from meeting_protocol.models import Segment, Speaker, Transcript
from meeting_protocol.transcription.merge import merge_transcripts


def _seg(
    id: int, start: float, end: float, text: str = "x", speaker_id: str | None = None
) -> Segment:
    return Segment(id=id, start=start, end=end, text=text, speaker_id=speaker_id)


def _transcript(segments: list[Segment], source: str = "clip.wav") -> Transcript:
    duration = max((s.end for s in segments), default=0.0)
    return Transcript(segments=segments, duration=duration, source_file=source)


# ── single-transcript pass-through ───────────────────────────────────────────


def test_single_transcript_returned_unchanged() -> None:
    t = _transcript([_seg(1, 0.0, 5.0, "Hello")], "clip1.wav")
    result = merge_transcripts([t])
    assert result is t


# ── two-transcript merge ──────────────────────────────────────────────────────


def test_two_transcripts_segment_count() -> None:
    t1 = _transcript([_seg(1, 0.0, 5.0), _seg(2, 5.5, 10.0)])
    t2 = _transcript([_seg(1, 0.0, 3.0), _seg(2, 3.5, 7.0)])
    result = merge_transcripts([t1, t2])
    assert len(result.segments) == 4


def test_first_clip_segments_unchanged() -> None:
    t1 = _transcript([_seg(1, 0.0, 5.0, "A"), _seg(2, 5.5, 10.0, "B")])
    t2 = _transcript([_seg(1, 0.0, 3.0, "C")])
    result = merge_transcripts([t1, t2])
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 5.0
    assert result.segments[1].start == 5.5
    assert result.segments[1].end == 10.0


def test_second_clip_start_offset_by_first_clip_max_end() -> None:
    """Clip2 segment originally at 0.0 must land at clip1.max_end (10.0)."""
    t1 = _transcript([_seg(1, 0.0, 5.0), _seg(2, 5.5, 10.0)])
    t2 = _transcript([_seg(1, 0.0, 3.0), _seg(2, 3.5, 7.0)])
    result = merge_transcripts([t1, t2])
    assert result.segments[2].start == 10.0  # 0.0 + 10.0
    assert result.segments[2].end == 13.0  # 3.0 + 10.0
    assert result.segments[3].start == 13.5  # 3.5 + 10.0
    assert result.segments[3].end == 17.0  # 7.0 + 10.0


# ── three-transcript merge: offset accumulates ───────────────────────────────


def test_three_clips_offset_accumulates() -> None:
    """Third clip offset = max_end(clip1) + duration(clip2)."""
    t1 = _transcript([_seg(1, 0.0, 10.0)])  # ends at 10.0
    t2 = _transcript([_seg(1, 0.0, 5.0)])  # ends at 5.0 → offset next by 15.0
    t3 = _transcript([_seg(1, 0.0, 8.0)])  # starts at 15.0, ends at 23.0
    result = merge_transcripts([t1, t2, t3])
    assert result.segments[2].start == 15.0
    assert result.segments[2].end == 23.0


# ── id re-numbering ───────────────────────────────────────────────────────────


def test_segment_ids_renumbered_sequentially() -> None:
    t1 = _transcript([_seg(1, 0.0, 5.0), _seg(2, 5.0, 10.0)])
    t2 = _transcript([_seg(1, 0.0, 3.0), _seg(2, 3.0, 6.0)])
    result = merge_transcripts([t1, t2])
    assert [s.id for s in result.segments] == [1, 2, 3, 4]


# ── duration ─────────────────────────────────────────────────────────────────


def test_merged_duration_is_total_end_time() -> None:
    t1 = _transcript([_seg(1, 0.0, 10.0)])
    t2 = _transcript([_seg(1, 0.0, 5.0)])
    result = merge_transcripts([t1, t2])
    assert result.duration == 15.0


# ── source_file ───────────────────────────────────────────────────────────────


def test_merged_source_file_contains_all_clip_paths() -> None:
    t1 = _transcript([], "clip1.wav")
    t2 = _transcript([], "clip2.wav")
    result = merge_transcripts([t1, t2])
    assert "clip1.wav" in result.source_file
    assert "clip2.wav" in result.source_file


# ── segment fields preserved ─────────────────────────────────────────────────


def test_text_and_speaker_id_preserved_through_merge() -> None:
    t1 = _transcript([_seg(1, 0.0, 5.0, "Hello", speaker_id="S1")])
    t2 = _transcript([_seg(1, 0.0, 3.0, "World", speaker_id="S2")])
    result = merge_transcripts([t1, t2])
    assert result.segments[0].text == "Hello"
    assert result.segments[0].speaker_id == "S1"
    assert result.segments[1].text == "World"
    assert result.segments[1].speaker_id == "S2"


# ── speakers merged ───────────────────────────────────────────────────────────


def test_speakers_from_all_clips_appear_in_merged_transcript() -> None:
    t1 = Transcript(
        segments=[],
        duration=0.0,
        source_file="c1.wav",
        speakers=[Speaker(id="S1", name="Yogev")],
    )
    t2 = Transcript(
        segments=[],
        duration=0.0,
        source_file="c2.wav",
        speakers=[Speaker(id="S2", name="Tom")],
    )
    result = merge_transcripts([t1, t2])
    ids = {sp.id for sp in result.speakers}
    assert "S1" in ids
    assert "S2" in ids


def test_duplicate_speakers_not_repeated() -> None:
    sp = Speaker(id="S1", name="Yogev")
    t1 = Transcript(segments=[], duration=0.0, source_file="c1.wav", speakers=[sp])
    t2 = Transcript(segments=[], duration=0.0, source_file="c2.wav", speakers=[sp])
    result = merge_transcripts([t1, t2])
    assert len([s for s in result.speakers if s.id == "S1"]) == 1


# ── empty segments edge case ─────────────────────────────────────────────────


def test_merge_handles_clip_with_no_segments() -> None:
    t1 = _transcript([_seg(1, 0.0, 5.0)], "c1.wav")
    t2 = _transcript([], "c2.wav")
    t3 = _transcript([_seg(1, 0.0, 3.0)], "c3.wav")
    result = merge_transcripts([t1, t2, t3])
    # t2 has no segments; offset for t3 should still be t1's max_end (5.0)
    assert result.segments[1].start == 5.0
    assert result.segments[1].end == 8.0
