from pathlib import Path

from meeting_protocol.models import Segment, Speaker, Transcript
from meeting_protocol.transcription.base import TranscriptionProvider


class MockTranscriptionProvider(TranscriptionProvider):
    def transcribe(self, audio_path: Path) -> Transcript:
        speakers = [
            Speaker(id="S1", name="Yogev"),
            Speaker(id="S2", name="Tom"),
        ]
        segments = [
            Segment(
                id=1,
                start=0.0,
                end=5.0,
                text="Let's start the meeting.",
                speaker_id="S1",
            ),
            Segment(
                id=2,
                start=5.5,
                end=12.0,
                text="We decided to proceed with the plan.",
                speaker_id="S2",
            ),
            Segment(
                id=3,
                start=12.5,
                end=20.0,
                text="I'll follow up on the action items.",
                speaker_id="S1",
            ),
        ]
        duration = max(seg.end for seg in segments)
        return Transcript(
            segments=segments,
            duration=duration,
            source_file=str(audio_path),
            speakers=speakers,
        )
