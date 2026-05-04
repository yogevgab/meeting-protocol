from abc import ABC, abstractmethod
from pathlib import Path

from meeting_protocol.models import Transcript


class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript: ...
