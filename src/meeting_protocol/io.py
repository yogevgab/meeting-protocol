import json
from pathlib import Path

from meeting_protocol.models import Transcript


def load_transcript(path: Path) -> Transcript:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Transcript.from_dict(data)


def save_transcript(transcript: Transcript, path: Path) -> None:
    Path(path).write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
