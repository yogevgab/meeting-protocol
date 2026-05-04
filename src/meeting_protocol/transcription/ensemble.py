from dataclasses import dataclass
from pathlib import Path

from meeting_protocol.models import Segment, Transcript
from meeting_protocol.transcription.base import TranscriptionProvider


@dataclass
class EnsembleTranscript:
    primary: Transcript
    secondary: Transcript | None
    secondary_texts: list[str | None]


def _iou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0.0:
        return 0.0
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0.0:
        return 0.0
    return inter / union


def align_segments(
    primary: list[Segment],
    secondary: list[Segment],
    iou_threshold: float = 0.3,
) -> list[str | None]:
    """For each primary segment, return the secondary text whose interval has
    the highest IoU with it, provided that IoU >= iou_threshold; otherwise None."""
    if not secondary:
        return [None] * len(primary)
    aligned: list[str | None] = []
    for p in primary:
        best_iou = 0.0
        best_text: str | None = None
        for s in secondary:
            score = _iou(p, s)
            if score > best_iou:
                best_iou = score
                best_text = s.text
        aligned.append(best_text if best_iou >= iou_threshold else None)
    return aligned


def transcribe_ensemble(
    primary_provider: TranscriptionProvider,
    secondary_provider: TranscriptionProvider | None,
    audio_path: Path,
    iou_threshold: float = 0.3,
) -> EnsembleTranscript:
    """Run primary (and optionally secondary) transcription and align secondary
    segments to primary by temporal IoU."""
    primary = primary_provider.transcribe(audio_path)
    if secondary_provider is None:
        return EnsembleTranscript(
            primary=primary,
            secondary=None,
            secondary_texts=[None] * len(primary.segments),
        )
    secondary = secondary_provider.transcribe(audio_path)
    secondary_texts = align_segments(primary.segments, secondary.segments, iou_threshold)
    return EnsembleTranscript(
        primary=primary,
        secondary=secondary,
        secondary_texts=secondary_texts,
    )
