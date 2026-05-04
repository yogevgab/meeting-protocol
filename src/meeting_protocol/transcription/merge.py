from dataclasses import replace as dc_replace

from meeting_protocol.models import Segment, Speaker, Transcript


def merge_transcripts(transcripts: list[Transcript]) -> Transcript:
    """Stitch multiple per-clip Transcripts into one, offsetting later timestamps.

    Each clip's segments are offset by the max segment end-time of all preceding
    clips so that the merged timeline is strictly non-overlapping and in order.
    Segment ids are renumbered 1..N across the merged result.
    """
    if not transcripts:
        raise ValueError("at least one Transcript is required")
    if len(transcripts) == 1:
        return transcripts[0]

    merged: list[Segment] = []
    seen_speaker_ids: set[str] = set()
    all_speakers: list[Speaker] = []
    offset = 0.0
    new_id = 1

    for transcript in transcripts:
        for seg in transcript.segments:
            merged.append(
                dc_replace(seg, id=new_id, start=seg.start + offset, end=seg.end + offset)
            )
            new_id += 1

        # Advance offset by the highest end-time in this clip's segments.
        if transcript.segments:
            offset = max(s.end for s in transcript.segments) + offset

        for sp in transcript.speakers:
            if sp.id not in seen_speaker_ids:
                seen_speaker_ids.add(sp.id)
                all_speakers.append(sp)

    duration = max((s.end for s in merged), default=0.0)
    source_file = "; ".join(t.source_file for t in transcripts)

    return Transcript(
        segments=merged,
        duration=duration,
        source_file=source_file,
        speakers=all_speakers,
        language=transcripts[0].language,
    )
