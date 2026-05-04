from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Self


class Language(StrEnum):
    HEBREW = "he"
    ENGLISH = "en"
    MIXED = "mixed"


class ActionStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker_id: str | None = None
    language: Language | None = None
    confidence: float | None = None


@dataclass
class Speaker:
    id: str
    name: str
    label: str | None = None


@dataclass
class Transcript:
    segments: list[Segment]
    duration: float
    source_file: str
    recorded_at: datetime | None = None
    speakers: list[Speaker] = field(default_factory=list)
    language: Language | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.recorded_at:
            d["recorded_at"] = self.recorded_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        recorded_at: datetime | None = None
        if d.get("recorded_at"):
            recorded_at = datetime.fromisoformat(d["recorded_at"])
        segments = [
            Segment(
                **{
                    **s,
                    "language": Language(s["language"]) if s.get("language") else None,
                }
            )
            for s in d["segments"]
        ]
        speakers = [Speaker(**sp) for sp in d.get("speakers", [])]
        language = Language(d["language"]) if d.get("language") else None
        return cls(
            segments=segments,
            duration=d["duration"],
            source_file=d["source_file"],
            recorded_at=recorded_at,
            speakers=speakers,
            language=language,
        )


@dataclass
class ActionItem:
    description: str
    owner: str | None = None
    due_date: str | None = None
    status: ActionStatus = ActionStatus.OPEN
    segment_id: int | None = None


@dataclass
class Decision:
    description: str
    context: str | None = None
    segment_id: int | None = None


@dataclass
class Highlight:
    text: str
    highlight_type: str  # "quote" | "idea" | "content_opportunity"
    segment_id: int | None = None


@dataclass
class Protocol:
    title: str
    date: datetime
    participants: list[str]
    source_file: str
    duration: float
    status: str = "draft"
    tldr: str = ""
    agenda_topics: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    business_ideas: list[Highlight] = field(default_factory=list)
    content_opportunities: list[Highlight] = field(default_factory=list)
    transcript: Transcript | None = None
