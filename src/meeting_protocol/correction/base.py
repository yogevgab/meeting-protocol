from abc import ABC, abstractmethod

from meeting_protocol.models import Transcript


class TranscriptCorrector(ABC):
    @abstractmethod
    def correct(
        self,
        transcript: Transcript,
        secondary_texts: list[str | None] | None = None,
    ) -> Transcript: ...
