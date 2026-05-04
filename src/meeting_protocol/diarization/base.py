from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerInterval:
    start: float
    end: float
    label: str  # anonymous label from the diarizer, e.g. "SPEAKER_00"


class Diarizer(ABC):
    @abstractmethod
    def diarize(self, audio_path: Path) -> list[SpeakerInterval]: ...
