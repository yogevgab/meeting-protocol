from dataclasses import replace

from meeting_protocol.diarization.base import SpeakerInterval
from meeting_protocol.models import Segment, Speaker


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _dominant_label(seg: Segment, intervals: list[SpeakerInterval]) -> str | None:
    """Return the label whose total temporal overlap with the segment is largest,
    or None if no interval overlaps at all. Sums overlap per label so a segment
    spanning multiple short turns of the same speaker still attributes correctly.
    Overlapping intervals from different labels are scored independently, so
    overlapped speech still attributes to whichever speaker dominated.
    Tie-break is on label name (lex ascending) for run-to-run stability."""
    totals: dict[str, float] = {}
    for iv in intervals:
        ov = _overlap(seg.start, seg.end, iv.start, iv.end)
        if ov > 0.0:
            totals[iv.label] = totals.get(iv.label, 0.0) + ov
    if not totals:
        return None
    # Negate the score so that highest score wins, then break ties by label name ascending.
    return min(totals.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def assign_speakers(
    segments: list[Segment],
    intervals: list[SpeakerInterval],
    speaker_map: dict[str, str] | None = None,
) -> tuple[list[Segment], list[Speaker]]:
    """Assign each segment to its dominant speaker label and build the
    minimal Speaker list of labels that were actually used.

    `speaker_map` translates anonymous labels (e.g. "SPEAKER_00") to display
    names ("Yogev"). Untranslated labels keep their anonymous form as the name."""
    name_for: dict[str, str] = {}
    new_segments: list[Segment] = []
    for seg in segments:
        label = _dominant_label(seg, intervals)
        if label is None:
            new_segments.append(seg)
            continue
        speaker_id = label
        display_name = (speaker_map or {}).get(label, label)
        name_for[speaker_id] = display_name
        new_segments.append(replace(seg, speaker_id=speaker_id))

    speakers = [Speaker(id=sid, name=name) for sid, name in name_for.items()]
    return new_segments, speakers
