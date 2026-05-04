from pathlib import Path

from meeting_protocol.models import Segment, Transcript
from meeting_protocol.transcription.base import TranscriptionProvider
from meeting_protocol.transcription.ensemble import (
    EnsembleTranscript,
    align_segments,
    transcribe_ensemble,
)


def _seg(seg_id: int, start: float, end: float, text: str) -> Segment:
    return Segment(id=seg_id, start=start, end=end, text=text)


# ── pure alignment ────────────────────────────────────────────────────────────


def test_align_returns_one_entry_per_primary_segment() -> None:
    primary = [_seg(1, 0.0, 1.0, "a"), _seg(2, 1.0, 2.0, "b")]
    secondary = [_seg(1, 0.0, 1.0, "x")]

    aligned = align_segments(primary, secondary)

    assert len(aligned) == 2


def test_align_perfect_overlap_returns_secondary_text() -> None:
    primary = [_seg(1, 0.0, 5.0, "primary")]
    secondary = [_seg(1, 0.0, 5.0, "secondary")]

    assert align_segments(primary, secondary) == ["secondary"]


def test_align_partial_overlap_above_threshold_returns_secondary_text() -> None:
    # IoU = 4 / 6 ≈ 0.67
    primary = [_seg(1, 0.0, 5.0, "primary")]
    secondary = [_seg(1, 1.0, 6.0, "secondary")]

    assert align_segments(primary, secondary, iou_threshold=0.3) == ["secondary"]


def test_align_low_overlap_below_threshold_returns_none() -> None:
    # IoU = 1 / 9 ≈ 0.11, below 0.3 threshold
    primary = [_seg(1, 0.0, 5.0, "primary")]
    secondary = [_seg(1, 4.0, 9.0, "secondary")]

    assert align_segments(primary, secondary, iou_threshold=0.3) == [None]


def test_align_no_overlap_returns_none() -> None:
    primary = [_seg(1, 0.0, 1.0, "primary")]
    secondary = [_seg(1, 5.0, 6.0, "secondary")]

    assert align_segments(primary, secondary) == [None]


def test_align_picks_highest_iou_among_candidates() -> None:
    primary = [_seg(1, 0.0, 10.0, "p")]
    secondary = [
        _seg(1, 0.0, 5.0, "early"),  # IoU = 5/10 = 0.5
        _seg(2, 0.0, 9.0, "best"),   # IoU = 9/10 = 0.9
        _seg(3, 0.0, 1.0, "tiny"),   # IoU = 1/10 = 0.1
    ]

    assert align_segments(primary, secondary) == ["best"]


def test_align_empty_secondary_returns_all_none() -> None:
    primary = [_seg(1, 0.0, 1.0, "a"), _seg(2, 1.0, 2.0, "b")]

    assert align_segments(primary, []) == [None, None]


def test_align_empty_primary_returns_empty() -> None:
    assert align_segments([], [_seg(1, 0.0, 1.0, "x")]) == []


def test_align_mismatched_counts_handled_per_segment() -> None:
    primary = [
        _seg(1, 0.0, 1.0, "a"),
        _seg(2, 1.0, 2.0, "b"),
        _seg(3, 2.0, 3.0, "c"),
    ]
    secondary = [_seg(1, 0.0, 1.0, "X")]

    assert align_segments(primary, secondary) == ["X", None, None]


# ── orchestrator ──────────────────────────────────────────────────────────────


class _StubProvider(TranscriptionProvider):
    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript

    def transcribe(self, audio_path: Path) -> Transcript:
        return self._transcript


def test_transcribe_ensemble_without_secondary_returns_all_none(tmp_path: Path) -> None:
    primary_t = Transcript(
        segments=[_seg(1, 0.0, 1.0, "p")],
        duration=1.0,
        source_file=str(tmp_path / "a.wav"),
    )
    primary = _StubProvider(primary_t)

    result = transcribe_ensemble(primary, None, tmp_path / "a.wav")

    assert isinstance(result, EnsembleTranscript)
    assert result.secondary is None
    assert result.secondary_texts == [None]
    assert result.primary is primary_t


def test_transcribe_ensemble_with_secondary_aligns(tmp_path: Path) -> None:
    primary_t = Transcript(
        segments=[_seg(1, 0.0, 5.0, "primary")],
        duration=5.0,
        source_file=str(tmp_path / "a.wav"),
    )
    secondary_t = Transcript(
        segments=[_seg(1, 0.0, 5.0, "secondary")],
        duration=5.0,
        source_file=str(tmp_path / "a.wav"),
    )
    primary = _StubProvider(primary_t)
    secondary = _StubProvider(secondary_t)

    result = transcribe_ensemble(primary, secondary, tmp_path / "a.wav")

    assert result.secondary is secondary_t
    assert result.secondary_texts == ["secondary"]
