# Meeting Protocol — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first Python CLI toolkit that transcribes Hebrew/English meetings, identifies speakers, and produces Obsidian-friendly Markdown protocols with decisions, action items, and highlights.

**Architecture:** Adapter-based providers for transcription (mock → whisper.cpp → faster-whisper) and diarization (manual → pyannote), thin domain models in `models.py`, heuristic extractors for Hebrew/English, and a generator that assembles everything into a `Protocol`. CLI built with Typer dispatches to providers and writers.

**Tech Stack:** Python 3.11+, Typer, Rich, pytest, ruff, mypy, setuptools `src` layout, GitHub Actions CI.

---

## File Map

```
meeting-protocol/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   ├── PLAN.md
│   └── BACKENDS.md              # whisper.cpp / faster-whisper install guide
├── fixtures/
│   └── sample_transcript.json   # Hebrew/English fixture for tests
├── src/
│   └── meeting_protocol/
│       ├── __init__.py
│       ├── models.py             # All typed domain models + JSON round-trip
│       ├── transcription/
│       │   ├── __init__.py
│       │   ├── base.py           # TranscriptionProvider ABC
│       │   ├── mock.py           # MockTranscriptionProvider (no deps)
│       │   └── whisper_cpp.py    # whisper.cpp CLI adapter
│       ├── diarization/
│       │   ├── __init__.py
│       │   ├── base.py           # DiarizationProvider ABC
│       │   └── manual.py         # ManualDiarizationProvider (speaker_map dict)
│       ├── protocol/
│       │   ├── __init__.py
│       │   ├── extractors.py     # Regex heuristics for Hebrew + English
│       │   └── generator.py      # ProtocolGenerator: Transcript → Protocol
│       ├── outputs/
│       │   ├── __init__.py
│       │   └── writers.py        # All output format writers
│       └── cli/
│           ├── __init__.py
│           └── main.py           # Typer app: transcribe / generate-protocol / run
├── tests/
│   ├── conftest.py               # Shared fixtures (sample segments, transcript)
│   ├── test_models.py
│   ├── test_extractors.py
│   ├── test_generator.py
│   ├── test_writers.py
│   └── test_cli.py
├── pyproject.toml
├── README.md
├── CLAUDE.md
└── LICENSE
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.github/workflows/ci.yml`
- Create: `src/meeting_protocol/__init__.py`
- Create: `src/meeting_protocol/transcription/__init__.py`
- Create: `src/meeting_protocol/diarization/__init__.py`
- Create: `src/meeting_protocol/protocol/__init__.py`
- Create: `src/meeting_protocol/outputs/__init__.py`
- Create: `src/meeting_protocol/cli/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=70", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "meeting-protocol"
version = "0.1.0"
description = "Local-first meeting transcription, diarization, and protocol generation for Hebrew + English"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "rich>=13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.4",
    "mypy>=1.10",
]
whisper-cpp = []
faster-whisper = ["faster-whisper>=1.0"]
pyannote = ["pyannote.audio>=3.1"]

[project.scripts]
meeting-protocol = "meeting_protocol.cli.main:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
mypy_path = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package directories and empty `__init__.py` files**

Run:
```bash
mkdir -p src/meeting_protocol/{transcription,diarization,protocol,outputs,cli}
mkdir -p tests fixtures
touch src/meeting_protocol/__init__.py
touch src/meeting_protocol/transcription/__init__.py
touch src/meeting_protocol/diarization/__init__.py
touch src/meeting_protocol/protocol/__init__.py
touch src/meeting_protocol/outputs/__init__.py
touch src/meeting_protocol/cli/__init__.py
touch tests/__init__.py
```

`src/meeting_protocol/__init__.py`:
```python
__version__ = "0.1.0"
```

All other `__init__.py` files are empty.

- [ ] **Step 3: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Type check
        run: mypy src/
      - name: Test
        run: pytest tests/ -v --tb=short --cov=meeting_protocol --cov-report=term-missing
```

- [ ] **Step 4: Install the package locally**

Run:
```bash
pip install -e ".[dev]"
```

Expected: no errors, `meeting-protocol` command available.

- [ ] **Step 5: Verify the scaffold is importable**

Run:
```bash
python -c "import meeting_protocol; print(meeting_protocol.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .github/ src/ tests/ fixtures/
git commit -m "feat: project scaffold with packaging, CI, and empty modules"
```

---

## Task 2: Domain Models

**Files:**
- Create: `src/meeting_protocol/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

`tests/test_models.py`:
```python
from datetime import datetime
from meeting_protocol.models import (
    Segment, Transcript, Speaker, Language,
    ActionItem, ActionStatus, Decision, Highlight, Protocol,
)


def test_language_enum_values() -> None:
    assert Language.HEBREW.value == "he"
    assert Language.ENGLISH.value == "en"
    assert Language.MIXED.value == "mixed"


def test_segment_defaults() -> None:
    seg = Segment(id=1, start=0.0, end=5.0, text="hello")
    assert seg.speaker_id is None
    assert seg.language is None
    assert seg.confidence is None


def test_transcript_to_dict_round_trip() -> None:
    transcript = Transcript(
        segments=[
            Segment(id=1, start=0.0, end=5.2, text="שלום", speaker_id="S1", language=Language.HEBREW, confidence=0.95),
            Segment(id=2, start=5.5, end=10.0, text="Hello", speaker_id="S2", language=Language.ENGLISH, confidence=0.97),
        ],
        duration=10.0,
        source_file="meeting.mp4",
        recorded_at=datetime(2024, 1, 15, 10, 0, 0),
        speakers=[Speaker(id="S1", name="Yogev"), Speaker(id="S2", name="Tom")],
        language=Language.MIXED,
    )
    d = transcript.to_dict()
    reconstructed = Transcript.from_dict(d)
    assert reconstructed.source_file == "meeting.mp4"
    assert reconstructed.duration == 10.0
    assert len(reconstructed.segments) == 2
    assert reconstructed.segments[0].text == "שלום"
    assert reconstructed.segments[0].language == Language.HEBREW
    assert reconstructed.recorded_at == datetime(2024, 1, 15, 10, 0, 0)
    assert len(reconstructed.speakers) == 2
    assert reconstructed.speakers[0].name == "Yogev"


def test_transcript_from_dict_no_recorded_at() -> None:
    d = {
        "segments": [{"id": 1, "start": 0.0, "end": 5.0, "text": "hi", "speaker_id": None, "language": None, "confidence": None}],
        "duration": 5.0,
        "source_file": "x.mp4",
        "recorded_at": None,
        "speakers": [],
        "language": None,
    }
    t = Transcript.from_dict(d)
    assert t.recorded_at is None


def test_action_item_default_status() -> None:
    a = ActionItem(description="do the thing")
    assert a.status == ActionStatus.OPEN


def test_protocol_fields() -> None:
    p = Protocol(
        title="Sprint Review",
        date=datetime(2024, 1, 15),
        participants=["Yogev", "Tom"],
        source_file="meeting.mp4",
        duration=3600.0,
    )
    assert p.status == "draft"
    assert p.decisions == []
    assert p.action_items == []
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```bash
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — models don't exist yet.

- [ ] **Step 3: Implement `src/meeting_protocol/models.py`**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Language(str, Enum):
    HEBREW = "he"
    ENGLISH = "en"
    MIXED = "mixed"


class ActionStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    language: Optional[Language] = None
    confidence: Optional[float] = None


@dataclass
class Speaker:
    id: str
    name: str
    label: Optional[str] = None


@dataclass
class Transcript:
    segments: list[Segment]
    duration: float
    source_file: str
    recorded_at: Optional[datetime] = None
    speakers: list[Speaker] = field(default_factory=list)
    language: Optional[Language] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.recorded_at:
            d["recorded_at"] = self.recorded_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transcript:
        recorded_at: Optional[datetime] = None
        if d.get("recorded_at"):
            recorded_at = datetime.fromisoformat(d["recorded_at"])
        segments = [
            Segment(
                **{**s, "language": Language(s["language"]) if s.get("language") else None}
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
    owner: Optional[str] = None
    due_date: Optional[str] = None
    status: ActionStatus = ActionStatus.OPEN
    segment_id: Optional[int] = None


@dataclass
class Decision:
    description: str
    context: Optional[str] = None
    segment_id: Optional[int] = None


@dataclass
class Highlight:
    text: str
    highlight_type: str  # "quote" | "idea" | "content_opportunity"
    segment_id: Optional[int] = None


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
    transcript: Optional[Transcript] = None
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:
```bash
pytest tests/test_models.py -v
```

Expected: all tests `PASSED`.

- [ ] **Step 5: Type check**

Run:
```bash
mypy src/meeting_protocol/models.py
```

Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/meeting_protocol/models.py tests/test_models.py
git commit -m "feat: add typed domain models with JSON round-trip"
```

---

## Task 3: Transcription Providers

**Files:**
- Create: `src/meeting_protocol/transcription/base.py`
- Create: `src/meeting_protocol/transcription/mock.py`
- Create: `src/meeting_protocol/transcription/whisper_cpp.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 1: Write failing tests**

`tests/test_transcription.py`:
```python
from meeting_protocol.transcription.mock import MockTranscriptionProvider
from meeting_protocol.models import Transcript, Segment, Language


def test_mock_provider_returns_transcript() -> None:
    provider = MockTranscriptionProvider()
    transcript = provider.transcribe("fake_audio.mp3")
    assert isinstance(transcript, Transcript)
    assert transcript.source_file == "fake_audio.mp3"
    assert len(transcript.segments) > 0


def test_mock_provider_uses_custom_segments() -> None:
    custom = [Segment(id=1, start=0.0, end=3.0, text="custom text")]
    provider = MockTranscriptionProvider(segments=custom)
    transcript = provider.transcribe("audio.mp4")
    assert transcript.segments[0].text == "custom text"
    assert transcript.duration == 3.0


def test_mock_provider_duration_matches_last_segment() -> None:
    segs = [
        Segment(id=1, start=0.0, end=5.0, text="a"),
        Segment(id=2, start=5.1, end=42.7, text="b"),
    ]
    provider = MockTranscriptionProvider(segments=segs)
    transcript = provider.transcribe("x.mp3")
    assert transcript.duration == 42.7
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```bash
pytest tests/test_transcription.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/meeting_protocol/transcription/base.py`**

```python
from abc import ABC, abstractmethod
from ..models import Transcript


class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> Transcript: ...
```

- [ ] **Step 4: Implement `src/meeting_protocol/transcription/mock.py`**

```python
from ..models import Language, Segment, Transcript
from .base import TranscriptionProvider

_DEFAULT_SEGMENTS = [
    Segment(id=1, start=0.0, end=5.0, text="Mock Hebrew segment: שלום עולם.", speaker_id="S1", language=Language.HEBREW),
    Segment(id=2, start=5.1, end=10.0, text="Mock English segment: hello world.", speaker_id="S2", language=Language.ENGLISH),
]


class MockTranscriptionProvider(TranscriptionProvider):
    def __init__(self, segments: list[Segment] | None = None) -> None:
        self._segments = segments if segments is not None else _DEFAULT_SEGMENTS

    def transcribe(self, audio_path: str) -> Transcript:
        return Transcript(
            segments=self._segments,
            duration=self._segments[-1].end if self._segments else 0.0,
            source_file=audio_path,
        )
```

- [ ] **Step 5: Implement `src/meeting_protocol/transcription/whisper_cpp.py`**

```python
import json
import subprocess
from pathlib import Path

from ..models import Segment, Transcript
from .base import TranscriptionProvider


class WhisperCppProvider(TranscriptionProvider):
    """Adapter for whisper.cpp CLI binary. See docs/BACKENDS.md for install instructions."""

    def __init__(self, model: str = "medium", language: str = "he") -> None:
        self.model = model
        self.language = language

    def transcribe(self, audio_path: str) -> Transcript:
        output_stem = str(Path(audio_path).with_suffix(""))
        subprocess.run(
            [
                "whisper-cpp",
                "--model", self.model,
                "--language", self.language,
                "--output-json",
                "--output-file", output_stem,
                audio_path,
            ],
            check=True,
        )
        raw = json.loads(Path(f"{output_stem}.json").read_text(encoding="utf-8"))
        segments = [
            Segment(
                id=i,
                start=seg["offsets"]["from"] / 1000.0,
                end=seg["offsets"]["to"] / 1000.0,
                text=seg["text"].strip(),
            )
            for i, seg in enumerate(raw.get("transcription", []))
        ]
        return Transcript(
            segments=segments,
            duration=segments[-1].end if segments else 0.0,
            source_file=audio_path,
        )
```

- [ ] **Step 6: Run tests and confirm they pass**

Run:
```bash
pytest tests/test_transcription.py -v
```

Expected: all `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add src/meeting_protocol/transcription/ tests/test_transcription.py
git commit -m "feat: add transcription provider abstraction, mock, and whisper.cpp adapter"
```

---

## Task 4: Diarization Providers

**Files:**
- Create: `src/meeting_protocol/diarization/base.py`
- Create: `src/meeting_protocol/diarization/manual.py`
- Create: `tests/test_diarization.py`

- [ ] **Step 1: Write failing tests**

`tests/test_diarization.py`:
```python
from meeting_protocol.diarization.manual import ManualDiarizationProvider
from meeting_protocol.models import Segment, Transcript, Speaker


def test_manual_diarization_sets_speakers(sample_transcript: Transcript) -> None:
    provider = ManualDiarizationProvider(speaker_map={"S1": "Yogev", "S2": "Tom"})
    result = provider.diarize(sample_transcript)
    speaker_names = {sp.name for sp in result.speakers}
    assert "Yogev" in speaker_names
    assert "Tom" in speaker_names


def test_manual_diarization_returns_transcript(sample_transcript: Transcript) -> None:
    provider = ManualDiarizationProvider(speaker_map={"S1": "Yogev"})
    result = provider.diarize(sample_transcript)
    assert isinstance(result, Transcript)
    assert len(result.segments) == len(sample_transcript.segments)


def test_manual_diarization_preserves_segment_speaker_ids(sample_transcript: Transcript) -> None:
    provider = ManualDiarizationProvider(speaker_map={"S1": "Yogev", "S2": "Tom"})
    result = provider.diarize(sample_transcript)
    assert result.segments[0].speaker_id == "S1"
    assert result.segments[1].speaker_id == "S2"
```

Note: `sample_transcript` fixture is defined in `tests/conftest.py` (Task 9).

- [ ] **Step 2: Create `tests/conftest.py` with shared fixtures**

```python
from datetime import datetime
import pytest
from meeting_protocol.models import Language, Segment, Speaker, Transcript


@pytest.fixture
def sample_segments() -> list[Segment]:
    return [
        Segment(id=1, start=0.0, end=5.2, text="אז בואו נדון על הפיצ'ר החדש שרצינו לבנות.", speaker_id="S1", language=Language.HEBREW, confidence=0.95),
        Segment(id=2, start=5.5, end=12.3, text="Right, we decided last week to build the analytics dashboard first.", speaker_id="S2", language=Language.ENGLISH, confidence=0.97),
        Segment(id=3, start=12.8, end=20.1, text="נכון, הוחלט שנתחיל עם ה-dashboard. צריך לסיים את זה עד סוף החודש.", speaker_id="S1", language=Language.HEBREW, confidence=0.93),
        Segment(id=4, start=20.5, end=28.0, text="I'll take ownership of the frontend part. Tom will handle the backend API.", speaker_id="S2", language=Language.ENGLISH, confidence=0.96),
        Segment(id=5, start=28.3, end=38.5, text="רעיון עסקי - אולי כדאי לחשוב על subscription model עבור הכלי הזה.", speaker_id="S1", language=Language.MIXED, confidence=0.91),
    ]


@pytest.fixture
def sample_transcript(sample_segments: list[Segment]) -> Transcript:
    return Transcript(
        segments=sample_segments,
        duration=38.5,
        source_file="meeting_2024_01_15.mp4",
        recorded_at=datetime(2024, 1, 15, 10, 0, 0),
        speakers=[Speaker(id="S1", name="Yogev"), Speaker(id="S2", name="Tom")],
        language=Language.MIXED,
    )
```

- [ ] **Step 3: Run tests to confirm they fail**

Run:
```bash
pytest tests/test_diarization.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/meeting_protocol/diarization/base.py`**

```python
from abc import ABC, abstractmethod
from ..models import Transcript


class DiarizationProvider(ABC):
    @abstractmethod
    def diarize(self, transcript: Transcript) -> Transcript:
        """Return transcript with speaker_id set on each segment."""
        ...
```

- [ ] **Step 5: Implement `src/meeting_protocol/diarization/manual.py`**

```python
from ..models import Speaker, Transcript
from .base import DiarizationProvider


class ManualDiarizationProvider(DiarizationProvider):
    """Maps existing speaker_id placeholders to real names via a provided dict."""

    def __init__(self, speaker_map: dict[str, str]) -> None:
        self._speaker_map = speaker_map

    def diarize(self, transcript: Transcript) -> Transcript:
        transcript.speakers = [
            Speaker(id=sid, name=name) for sid, name in self._speaker_map.items()
        ]
        return transcript
```

- [ ] **Step 6: Run tests and confirm they pass**

Run:
```bash
pytest tests/test_diarization.py -v
```

Expected: all `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add src/meeting_protocol/diarization/ tests/conftest.py tests/test_diarization.py
git commit -m "feat: add diarization provider abstraction and manual speaker mapping"
```

---

## Task 5: Protocol Extractors

**Files:**
- Create: `src/meeting_protocol/protocol/extractors.py`
- Create: `tests/test_extractors.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors.py`:
```python
from meeting_protocol.models import Segment, Language
from meeting_protocol.protocol.extractors import (
    extract_action_items,
    extract_decisions,
    extract_highlights,
)


def test_extracts_hebrew_decision(sample_segments: list[Segment]) -> None:
    decisions = extract_decisions(sample_segments)
    assert any("הוחלט" in d.description for d in decisions)


def test_extracts_english_decision(sample_segments: list[Segment]) -> None:
    decisions = extract_decisions(sample_segments)
    assert any("decided" in d.description.lower() for d in decisions)


def test_decision_carries_segment_id(sample_segments: list[Segment]) -> None:
    decisions = extract_decisions(sample_segments)
    assert all(d.segment_id is not None for d in decisions)


def test_extracts_hebrew_action_item(sample_segments: list[Segment]) -> None:
    actions = extract_action_items(sample_segments)
    assert any("צריך" in a.description for a in actions)


def test_extracts_english_action_item(sample_segments: list[Segment]) -> None:
    actions = extract_action_items(sample_segments)
    assert any("I'll" in a.description for a in actions)


def test_action_item_owner_is_speaker_id(sample_segments: list[Segment]) -> None:
    actions = extract_action_items(sample_segments)
    assert all(a.owner is not None for a in actions)


def test_extracts_business_idea(sample_segments: list[Segment]) -> None:
    highlights = extract_highlights(sample_segments)
    ideas = [h for h in highlights if h.highlight_type == "idea"]
    assert len(ideas) >= 1
    assert any("רעיון עסקי" in h.text for h in ideas)


def test_no_false_positives_plain_segment() -> None:
    segments = [Segment(id=1, start=0.0, end=5.0, text="The weather is nice today.")]
    assert extract_decisions(segments) == []
    assert extract_action_items(segments) == []
    assert extract_highlights(segments) == []


def test_no_false_positives_hebrew_plain() -> None:
    segments = [Segment(id=1, start=0.0, end=5.0, text="הדיון היה מעניין.", language=Language.HEBREW)]
    assert extract_decisions(segments) == []
    assert extract_action_items(segments) == []
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```bash
pytest tests/test_extractors.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/meeting_protocol/protocol/extractors.py`**

```python
import re
from ..models import ActionItem, Decision, Highlight, Segment

_ACTION_HE = re.compile(
    r"(?:צריך|יש ל|אנחנו צריכים|נצטרך|תעשה|תדאג|נעשה|נבדוק|ניצור|ניקח|נכתוב|נשלח)",
    re.IGNORECASE,
)
_ACTION_EN = re.compile(
    r"(?:TODO|action item|action:|task:|need to|will do|I'll|we should|follow up|we need to)",
    re.IGNORECASE,
)
_DECISION_HE = re.compile(
    r"(?:הוחלט|נחליט|החלטנו|הסכמנו|מחליטים ש|הסכמה ש|הוחלט ש)",
    re.IGNORECASE,
)
_DECISION_EN = re.compile(
    r"(?:we decided|decided to|decision:|we agreed|resolved to|agreed to|we're going with)",
    re.IGNORECASE,
)
_IDEA = re.compile(
    r"(?:business idea|רעיון עסקי|what if we|we could build|opportunity|maybe we should consider|חשוב לשקול|רעיון ל)",
    re.IGNORECASE,
)
_CONTENT = re.compile(
    r"(?:content opportunity|post about|video idea|episode about|we could write|thread about|נוכל לכתוב|פוסט על)",
    re.IGNORECASE,
)


def extract_action_items(segments: list[Segment]) -> list[ActionItem]:
    return [
        ActionItem(description=seg.text.strip(), owner=seg.speaker_id, segment_id=seg.id)
        for seg in segments
        if _ACTION_HE.search(seg.text) or _ACTION_EN.search(seg.text)
    ]


def extract_decisions(segments: list[Segment]) -> list[Decision]:
    return [
        Decision(description=seg.text.strip(), segment_id=seg.id)
        for seg in segments
        if _DECISION_HE.search(seg.text) or _DECISION_EN.search(seg.text)
    ]


def extract_highlights(segments: list[Segment]) -> list[Highlight]:
    results: list[Highlight] = []
    for seg in segments:
        if _IDEA.search(seg.text):
            results.append(Highlight(text=seg.text.strip(), highlight_type="idea", segment_id=seg.id))
        elif _CONTENT.search(seg.text):
            results.append(Highlight(text=seg.text.strip(), highlight_type="content_opportunity", segment_id=seg.id))
    return results
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:
```bash
pytest tests/test_extractors.py -v
```

Expected: all `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/meeting_protocol/protocol/extractors.py tests/test_extractors.py
git commit -m "feat: add Hebrew/English heuristic extractors for actions, decisions, highlights"
```

---

## Task 6: Protocol Generator

**Files:**
- Create: `src/meeting_protocol/protocol/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_generator.py`:
```python
from datetime import datetime
from meeting_protocol.protocol.generator import ProtocolGenerator
from meeting_protocol.models import Protocol, Transcript


def test_generator_produces_protocol(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript, title="Test Meeting")
    assert isinstance(protocol, Protocol)
    assert protocol.title == "Test Meeting"
    assert protocol.duration == 38.5
    assert protocol.source_file == "meeting_2024_01_15.mp4"


def test_generator_uses_transcript_date(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript)
    assert protocol.date == datetime(2024, 1, 15, 10, 0, 0)


def test_generator_populates_participants(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript)
    assert "Yogev" in protocol.participants
    assert "Tom" in protocol.participants


def test_generator_extracts_decisions(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript)
    assert len(protocol.decisions) >= 1


def test_generator_extracts_action_items(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript)
    assert len(protocol.action_items) >= 1


def test_generator_extracts_business_ideas(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript)
    assert len(protocol.business_ideas) >= 1


def test_generator_default_title_uses_date(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript)
    assert "2024-01-15" in protocol.title


def test_generator_attaches_transcript(sample_transcript: Transcript) -> None:
    gen = ProtocolGenerator()
    protocol = gen.generate(sample_transcript)
    assert protocol.transcript is sample_transcript
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```bash
pytest tests/test_generator.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/meeting_protocol/protocol/generator.py`**

```python
from datetime import datetime, timezone
from typing import Optional

from ..models import Protocol, Transcript
from .extractors import extract_action_items, extract_decisions, extract_highlights


class ProtocolGenerator:
    def generate(self, transcript: Transcript, title: Optional[str] = None) -> Protocol:
        date = transcript.recorded_at or datetime.now(tz=timezone.utc).replace(tzinfo=None)
        participants = [sp.name for sp in transcript.speakers]
        action_items = extract_action_items(transcript.segments)
        decisions = extract_decisions(transcript.segments)
        highlights = extract_highlights(transcript.segments)
        return Protocol(
            title=title or f"Meeting {date.strftime('%Y-%m-%d')}",
            date=date,
            participants=participants,
            source_file=transcript.source_file,
            duration=transcript.duration,
            decisions=decisions,
            action_items=action_items,
            business_ideas=[h for h in highlights if h.highlight_type == "idea"],
            content_opportunities=[h for h in highlights if h.highlight_type == "content_opportunity"],
            transcript=transcript,
        )
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:
```bash
pytest tests/test_generator.py -v
```

Expected: all `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/meeting_protocol/protocol/generator.py tests/test_generator.py
git commit -m "feat: add ProtocolGenerator that assembles Transcript into Protocol"
```

---

## Task 7: Output Writers

**Files:**
- Create: `src/meeting_protocol/outputs/writers.py`
- Create: `tests/test_writers.py`

- [ ] **Step 1: Write failing tests**

`tests/test_writers.py`:
```python
import json
from pathlib import Path
from meeting_protocol.models import Transcript
from meeting_protocol.outputs.writers import (
    write_actions_markdown,
    write_highlights_markdown,
    write_protocol_markdown,
    write_transcript_json,
    write_transcript_markdown,
    write_transcript_srt,
)
from meeting_protocol.protocol.generator import ProtocolGenerator


def test_write_transcript_json_is_valid(sample_transcript: Transcript, tmp_path: Path) -> None:
    out = tmp_path / "transcript.json"
    write_transcript_json(sample_transcript, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["source_file"] == "meeting_2024_01_15.mp4"
    assert len(data["segments"]) == 5


def test_write_transcript_json_hebrew_preserved(sample_transcript: Transcript, tmp_path: Path) -> None:
    out = tmp_path / "transcript.json"
    write_transcript_json(sample_transcript, out)
    content = out.read_text(encoding="utf-8")
    assert "הפיצ'ר" in content


def test_write_transcript_markdown_has_speakers(sample_transcript: Transcript, tmp_path: Path) -> None:
    out = tmp_path / "transcript.md"
    write_transcript_markdown(sample_transcript, out)
    content = out.read_text(encoding="utf-8")
    assert "**S1**" in content
    assert "**S2**" in content


def test_write_transcript_markdown_has_timestamps(sample_transcript: Transcript, tmp_path: Path) -> None:
    out = tmp_path / "transcript.md"
    write_transcript_markdown(sample_transcript, out)
    content = out.read_text(encoding="utf-8")
    assert "[00:00:00]" in content


def test_write_transcript_srt_format(sample_transcript: Transcript, tmp_path: Path) -> None:
    out = tmp_path / "transcript.srt"
    write_transcript_srt(sample_transcript, out)
    content = out.read_text(encoding="utf-8")
    assert "-->" in content
    assert "00:00:00,000 --> 00:00:05,200" in content


def test_write_protocol_markdown_sections(sample_transcript: Transcript, tmp_path: Path) -> None:
    protocol = ProtocolGenerator().generate(sample_transcript, title="Test Meeting")
    out = tmp_path / "protocol.md"
    write_protocol_markdown(protocol, out)
    content = out.read_text(encoding="utf-8")
    assert "# Test Meeting" in content
    assert "Yogev" in content
    assert "Tom" in content


def test_write_protocol_markdown_has_decisions(sample_transcript: Transcript, tmp_path: Path) -> None:
    protocol = ProtocolGenerator().generate(sample_transcript)
    out = tmp_path / "protocol.md"
    write_protocol_markdown(protocol, out)
    content = out.read_text(encoding="utf-8")
    assert "## Decisions" in content


def test_write_actions_markdown_table(sample_transcript: Transcript, tmp_path: Path) -> None:
    protocol = ProtocolGenerator().generate(sample_transcript, title="Test")
    out = tmp_path / "actions.md"
    write_actions_markdown(protocol, out)
    content = out.read_text(encoding="utf-8")
    assert "# Action Items" in content
    assert "| Description |" in content


def test_write_highlights_markdown_ideas(sample_transcript: Transcript, tmp_path: Path) -> None:
    protocol = ProtocolGenerator().generate(sample_transcript)
    out = tmp_path / "highlights.md"
    write_highlights_markdown(protocol, out)
    content = out.read_text(encoding="utf-8")
    assert "## Business Ideas" in content
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```bash
pytest tests/test_writers.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/meeting_protocol/outputs/writers.py`**

```python
import json
from pathlib import Path

from ..models import Protocol, Transcript


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_srt_time(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000))
    return f"{_fmt_time(seconds)},{ms:03d}"


def write_transcript_json(transcript: Transcript, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_transcript_markdown(transcript: Transcript, output_path: Path) -> None:
    lines = [f"# Transcript: {transcript.source_file}\n\n"]
    if transcript.recorded_at:
        lines.append(f"**Date:** {transcript.recorded_at.strftime('%Y-%m-%d %H:%M')}\n\n")
    lines.append(f"**Duration:** {_fmt_time(transcript.duration)}\n\n---\n\n")
    for seg in transcript.segments:
        speaker = seg.speaker_id or "Speaker"
        lines.append(f"**{speaker}** [{_fmt_time(seg.start)}]: {seg.text}\n\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def write_transcript_srt(transcript: Transcript, output_path: Path) -> None:
    blocks: list[str] = []
    for seg in transcript.segments:
        prefix = f"[{seg.speaker_id}] " if seg.speaker_id else ""
        blocks.append(
            f"{seg.id}\n{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}\n{prefix}{seg.text}\n"
        )
    output_path.write_text("\n".join(blocks), encoding="utf-8")


def write_protocol_markdown(protocol: Protocol, output_path: Path) -> None:
    lines = [
        f"# {protocol.title}\n\n",
        f"| Field | Value |\n|---|---|\n",
        f"| **Date** | {protocol.date.strftime('%Y-%m-%d')} |\n",
        f"| **Participants** | {', '.join(protocol.participants)} |\n",
        f"| **Source** | `{protocol.source_file}` |\n",
        f"| **Duration** | {_fmt_time(protocol.duration)} |\n",
        f"| **Status** | {protocol.status} |\n\n",
    ]
    if protocol.tldr:
        lines += [f"## TL;DR\n\n{protocol.tldr}\n\n"]
    if protocol.decisions:
        lines.append("## Decisions\n\n")
        for d in protocol.decisions:
            lines.append(f"- {d.description}\n")
        lines.append("\n")
    if protocol.action_items:
        lines.append("## Action Items\n\n")
        lines.append("| Description | Owner | Due | Status |\n|---|---|---|---|\n")
        for a in protocol.action_items:
            lines.append(
                f"| {a.description} | {a.owner or ''} | {a.due_date or ''} | {a.status.value} |\n"
            )
        lines.append("\n")
    if protocol.risks:
        lines.append("## Risks & Open Questions\n\n")
        for r in protocol.risks:
            lines.append(f"- {r}\n")
        lines.append("\n")
    if protocol.business_ideas:
        lines.append("## Business Ideas\n\n")
        for h in protocol.business_ideas:
            lines.append(f"- {h.text}\n")
        lines.append("\n")
    if protocol.content_opportunities:
        lines.append("## Content Opportunities\n\n")
        for h in protocol.content_opportunities:
            lines.append(f"- {h.text}\n")
        lines.append("\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def write_actions_markdown(protocol: Protocol, output_path: Path) -> None:
    lines = [f"# Action Items: {protocol.title}\n\n"]
    lines.append("| # | Description | Owner | Due | Status |\n|---|---|---|---|---|\n")
    for i, a in enumerate(protocol.action_items, 1):
        lines.append(
            f"| {i} | {a.description} | {a.owner or ''} | {a.due_date or ''} | {a.status.value} |\n"
        )
    output_path.write_text("".join(lines), encoding="utf-8")


def write_highlights_markdown(protocol: Protocol, output_path: Path) -> None:
    lines = [f"# Highlights: {protocol.title}\n\n"]
    if protocol.business_ideas:
        lines.append("## Business Ideas\n\n")
        for h in protocol.business_ideas:
            lines.append(f"> {h.text}\n\n")
    if protocol.content_opportunities:
        lines.append("## Content Opportunities\n\n")
        for h in protocol.content_opportunities:
            lines.append(f"> {h.text}\n\n")
    output_path.write_text("".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:
```bash
pytest tests/test_writers.py -v
```

Expected: all `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/meeting_protocol/outputs/writers.py tests/test_writers.py
git commit -m "feat: add output writers for JSON, Markdown, SRT, protocol, actions, highlights"
```

---

## Task 8: CLI

**Files:**
- Create: `src/meeting_protocol/cli/main.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cli.py`:
```python
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from meeting_protocol.cli.main import app
from meeting_protocol.models import Transcript
from meeting_protocol.outputs.writers import write_transcript_json

runner = CliRunner()


def test_transcribe_mock_creates_all_outputs(tmp_path: Path) -> None:
    result = runner.invoke(app, [
        "transcribe", "fake_audio.mp3",
        "--output-dir", str(tmp_path),
        "--backend", "mock",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.json").exists()
    assert (tmp_path / "transcript.md").exists()
    assert (tmp_path / "transcript.srt").exists()


def test_transcribe_mock_json_is_valid(tmp_path: Path) -> None:
    runner.invoke(app, ["transcribe", "fake.mp4", "--output-dir", str(tmp_path), "--backend", "mock"])
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert "segments" in data
    assert "source_file" in data


def test_generate_protocol_creates_outputs(sample_transcript: Transcript, tmp_path: Path) -> None:
    json_path = tmp_path / "transcript.json"
    write_transcript_json(sample_transcript, json_path)
    result = runner.invoke(app, [
        "generate-protocol", str(json_path),
        "--output-dir", str(tmp_path),
        "--title", "Test Meeting",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "protocol.md").exists()
    assert (tmp_path / "actions.md").exists()
    assert (tmp_path / "highlights.md").exists()


def test_generate_protocol_markdown_has_title(sample_transcript: Transcript, tmp_path: Path) -> None:
    json_path = tmp_path / "transcript.json"
    write_transcript_json(sample_transcript, json_path)
    runner.invoke(app, ["generate-protocol", str(json_path), "--output-dir", str(tmp_path), "--title", "Sprint Review"])
    content = (tmp_path / "protocol.md").read_text(encoding="utf-8")
    assert "# Sprint Review" in content


def test_run_full_pipeline_creates_all_outputs(tmp_path: Path) -> None:
    result = runner.invoke(app, [
        "run", "fake_audio.mp3",
        "--output-dir", str(tmp_path),
        "--backend", "mock",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcript.json").exists()
    assert (tmp_path / "protocol.md").exists()
    assert (tmp_path / "actions.md").exists()
    assert (tmp_path / "highlights.md").exists()


def test_help_shows_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "generate-protocol" in result.output
    assert "run" in result.output
```

- [ ] **Step 2: Run tests to confirm they fail**

Run:
```bash
pytest tests/test_cli.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/meeting_protocol/cli/main.py`**

```python
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ..diarization.manual import ManualDiarizationProvider
from ..models import Transcript
from ..outputs import writers
from ..protocol.generator import ProtocolGenerator
from ..transcription.mock import MockTranscriptionProvider
from ..transcription.whisper_cpp import WhisperCppProvider

app = typer.Typer(help="Meeting protocol toolkit — transcription, diarization, and minutes.")
console = Console()


def _get_transcription_provider(
    backend: str, model: str, language: str
) -> MockTranscriptionProvider | WhisperCppProvider:
    if backend == "whisper-cpp":
        return WhisperCppProvider(model=model, language=language)
    return MockTranscriptionProvider()


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Path to audio/video file"),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o", help="Output directory"),
    backend: str = typer.Option("mock", "--backend", "-b", help="Transcription backend: mock | whisper-cpp"),
    model: str = typer.Option("medium", "--model", "-m", help="whisper.cpp model size"),
    language: str = typer.Option("he", "--language", "-l", help="Primary language code (he, en, auto)"),
) -> None:
    """Transcribe an audio or video file."""
    provider = _get_transcription_provider(backend, model, language)
    transcript = provider.transcribe(str(audio_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    writers.write_transcript_json(transcript, output_dir / "transcript.json")
    writers.write_transcript_markdown(transcript, output_dir / "transcript.md")
    writers.write_transcript_srt(transcript, output_dir / "transcript.srt")
    console.print(f"[green]Transcript written to {output_dir}[/green]")


@app.command(name="generate-protocol")
def generate_protocol(
    transcript_json: Path = typer.Argument(..., help="Path to transcript.json"),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Meeting title"),
) -> None:
    """Generate protocol files from a transcript JSON."""
    data = json.loads(transcript_json.read_text(encoding="utf-8"))
    transcript = Transcript.from_dict(data)
    protocol = ProtocolGenerator().generate(transcript, title=title)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers.write_protocol_markdown(protocol, output_dir / "protocol.md")
    writers.write_actions_markdown(protocol, output_dir / "actions.md")
    writers.write_highlights_markdown(protocol, output_dir / "highlights.md")
    console.print(f"[green]Protocol written to {output_dir}[/green]")


@app.command()
def run(
    audio_file: Path = typer.Argument(..., help="Path to audio/video file"),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o"),
    backend: str = typer.Option("mock", "--backend", "-b"),
    model: str = typer.Option("medium", "--model", "-m"),
    language: str = typer.Option("he", "--language", "-l"),
    speaker_map_file: Optional[Path] = typer.Option(None, "--speaker-map", help="JSON file mapping speaker IDs to names"),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
) -> None:
    """Full pipeline: transcribe → (diarize) → generate protocol."""
    provider = _get_transcription_provider(backend, model, language)
    transcript = provider.transcribe(str(audio_file))

    if speaker_map_file:
        speaker_map: dict[str, str] = json.loads(speaker_map_file.read_text(encoding="utf-8"))
        transcript = ManualDiarizationProvider(speaker_map=speaker_map).diarize(transcript)

    output_dir.mkdir(parents=True, exist_ok=True)
    writers.write_transcript_json(transcript, output_dir / "transcript.json")
    writers.write_transcript_markdown(transcript, output_dir / "transcript.md")
    writers.write_transcript_srt(transcript, output_dir / "transcript.srt")

    protocol = ProtocolGenerator().generate(transcript, title=title)
    writers.write_protocol_markdown(protocol, output_dir / "protocol.md")
    writers.write_actions_markdown(protocol, output_dir / "actions.md")
    writers.write_highlights_markdown(protocol, output_dir / "highlights.md")
    console.print(f"[green]All outputs written to {output_dir}[/green]")
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:
```bash
pytest tests/test_cli.py -v
```

Expected: all `PASSED`.

- [ ] **Step 5: Run the full test suite**

Run:
```bash
pytest tests/ -v --tb=short
```

Expected: all tests across all modules `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/meeting_protocol/cli/main.py tests/test_cli.py
git commit -m "feat: add Typer CLI with transcribe, generate-protocol, and run commands"
```

---

## Task 9: Sample Fixture Data

**Files:**
- Create: `fixtures/sample_transcript.json`
- Create: `fixtures/speaker_map.json`

- [ ] **Step 1: Create `fixtures/sample_transcript.json`**

This is a real Hebrew/English mixed transcript that all fixture-based tests use as canonical reference data.

```json
{
  "source_file": "meeting_2024_01_15.mp4",
  "duration": 38.5,
  "recorded_at": "2024-01-15T10:00:00",
  "language": "mixed",
  "speakers": [
    {"id": "S1", "name": "Yogev", "label": null},
    {"id": "S2", "name": "Tom", "label": null}
  ],
  "segments": [
    {
      "id": 1, "start": 0.0, "end": 5.2,
      "text": "אז בואו נדון על הפיצ'ר החדש שרצינו לבנות.",
      "speaker_id": "S1", "language": "he", "confidence": 0.95
    },
    {
      "id": 2, "start": 5.5, "end": 12.3,
      "text": "Right, we decided last week to build the analytics dashboard first.",
      "speaker_id": "S2", "language": "en", "confidence": 0.97
    },
    {
      "id": 3, "start": 12.8, "end": 20.1,
      "text": "נכון, הוחלט שנתחיל עם ה-dashboard. צריך לסיים את זה עד סוף החודש.",
      "speaker_id": "S1", "language": "he", "confidence": 0.93
    },
    {
      "id": 4, "start": 20.5, "end": 28.0,
      "text": "I'll take ownership of the frontend part. Tom will handle the backend API.",
      "speaker_id": "S2", "language": "en", "confidence": 0.96
    },
    {
      "id": 5, "start": 28.3, "end": 38.5,
      "text": "רעיון עסקי - אולי כדאי לחשוב על subscription model עבור הכלי הזה.",
      "speaker_id": "S1", "language": "mixed", "confidence": 0.91
    }
  ]
}
```

- [ ] **Step 2: Create `fixtures/speaker_map.json`**

```json
{
  "S1": "Yogev",
  "S2": "Tom"
}
```

- [ ] **Step 3: Smoke-test the fixture through the CLI**

Run:
```bash
mkdir -p /tmp/mp_test
cp fixtures/sample_transcript.json /tmp/mp_test/
meeting-protocol generate-protocol /tmp/mp_test/sample_transcript.json --output-dir /tmp/mp_test --title "Sample Meeting"
cat /tmp/mp_test/protocol.md
```

Expected: Markdown file with `# Sample Meeting`, decisions section containing "הוחלט" or "decided", action items section, business ideas section.

- [ ] **Step 4: Commit**

```bash
git add fixtures/
git commit -m "feat: add Hebrew/English sample fixture data for tests and demos"
```

---

## Task 10: Lint, Type Check, and Full CI Verification

**Files:** No new files — quality gate pass.

- [ ] **Step 1: Run ruff lint**

Run:
```bash
ruff check .
```

Expected: no output (zero violations). If violations appear, fix them before continuing.

- [ ] **Step 2: Run mypy**

Run:
```bash
mypy src/
```

Expected: `Success: no issues found in N source files`.

If mypy reports errors about missing return types or Optional handling, fix them in the relevant source files before proceeding.

- [ ] **Step 3: Run full test suite with coverage**

Run:
```bash
pytest tests/ -v --cov=meeting_protocol --cov-report=term-missing
```

Expected: all tests `PASSED`, coverage reported (aim for >80% on core modules).

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: address ruff/mypy findings"
```

(Skip this step if there were no fixes needed.)

---

## Task 11: README and BACKENDS.md

**Files:**
- Create: `README.md`
- Create: `docs/BACKENDS.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# meeting-protocol

Local-first toolkit for transcribing meetings and generating structured protocols.
Optimized for Hebrew + English mixed conversations on a Mac mini (or any local machine).

## Features

- Transcribe audio/video files (mock backend for tests; whisper.cpp/faster-whisper adapters for production)
- Speaker identification via manual mapping (pyannote integration planned)
- Generate Obsidian-friendly Markdown: `protocol.md`, `actions.md`, `highlights.md`, `transcript.md`
- Extracts decisions, action items, business ideas, and content opportunities from Hebrew/English text
- Export transcripts as JSON, Markdown, and SRT subtitles
- Privacy-first: all processing stays local, no cloud APIs required

## Requirements

- Python 3.11+
- For real transcription: [whisper.cpp](https://github.com/ggerganov/whisper.cpp) or [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — see [docs/BACKENDS.md](docs/BACKENDS.md)

## Installation

```bash
pip install meeting-protocol
```

Or from source:

```bash
git clone <repo-url>
cd meeting-protocol
pip install -e ".[dev]"
```

## Quickstart

```bash
# Transcribe (mock backend — no external tools needed)
meeting-protocol transcribe my_meeting.mp4 --backend mock --output-dir ./output

# Generate protocol from transcript JSON
meeting-protocol generate-protocol ./output/transcript.json --title "Sprint Review"

# Full pipeline (transcribe → generate protocol)
meeting-protocol run my_meeting.mp4 --backend whisper-cpp --language he --output-dir ./output

# Full pipeline with speaker mapping
meeting-protocol run my_meeting.mp4 --backend whisper-cpp --speaker-map fixtures/speaker_map.json --output-dir ./output
```

## Output files

| File | Description |
|---|---|
| `transcript.json` | Full transcript with timestamps, speakers, language |
| `transcript.md` | Human-readable transcript |
| `transcript.srt` | Subtitle file |
| `protocol.md` | Structured meeting minutes |
| `actions.md` | Action items table |
| `highlights.md` | Business ideas and content opportunities |

## Architecture

```
meeting_protocol/
├── models.py          # Typed domain models (Segment, Transcript, Protocol, …)
├── transcription/     # Provider abstraction: mock, whisper.cpp, faster-whisper
├── diarization/       # Provider abstraction: manual, pyannote (future)
├── protocol/          # Heuristic extractors + ProtocolGenerator
├── outputs/           # Writers for all output formats
└── cli/               # Typer CLI (transcribe / generate-protocol / run)
```

## Hebrew/English transcription quality

See [docs/BACKENDS.md](docs/BACKENDS.md) for model recommendations and tradeoffs for Hebrew.

**Short version:** Use `whisper-large-v3` or `whisper-large-v3-turbo` for best Hebrew quality. The `medium` model is a reasonable size/quality tradeoff for local inference on a Mac mini M-series.

## Development

```bash
pip install -e ".[dev]"
pytest tests/          # run tests
ruff check .           # lint
mypy src/              # type check
```

## Roadmap

- [ ] faster-whisper adapter for faster local inference
- [ ] pyannote.audio adapter for automatic speaker diarization
- [ ] LLM-based TL;DR and agenda extraction (optional, local Ollama backend)
- [ ] Obsidian vault folder structure helper
- [ ] VTT subtitle output
- [ ] Web UI (Gradio or Streamlit, optional extra)

## License

MIT
```

- [ ] **Step 2: Create `docs/BACKENDS.md`**

```markdown
# Transcription Backends

## whisper.cpp (recommended for local use)

### Installation (macOS / Apple Silicon)

```bash
brew install cmake
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_METAL=ON   # enables Metal GPU acceleration on Apple Silicon
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
cp build/bin/whisper-cpp /usr/local/bin/  # or add to PATH
```

### Download a model

```bash
# Inside the whisper.cpp repo
bash models/download-ggml-model.sh large-v3-turbo
```

Models are saved to `models/`. Reference the `.bin` path when running.

### Model recommendations for Hebrew + English

| Model | Size | Hebrew quality | Speed (M2 Mac) |
|---|---|---|---|
| `large-v3` | 3 GB | Excellent | ~0.5x realtime |
| `large-v3-turbo` | 1.5 GB | Very good | ~1x realtime |
| `medium` | 1.5 GB | Good | ~2x realtime |
| `small` | 450 MB | Acceptable | ~5x realtime |

For production use with mixed Hebrew/English, use `large-v3-turbo`.

### Usage with meeting-protocol

```bash
meeting-protocol run meeting.mp4 \
  --backend whisper-cpp \
  --model large-v3-turbo \
  --language he \
  --output-dir ./output
```

Whisper's `--language he` forces Hebrew but it still handles English segments.
Use `--language auto` for fully automatic detection (slightly slower).

---

## faster-whisper (optional extra)

Install:

```bash
pip install "meeting-protocol[faster-whisper]"
```

faster-whisper is a CTranslate2-based reimplementation of Whisper — typically 2-4× faster than whisper.cpp on CPU, with the same model weights. It does not yet have a CLI adapter in this project; contributions welcome.

---

## pyannote.audio (future — speaker diarization)

Automatic speaker diarization support is planned. It will require:

```bash
pip install "meeting-protocol[pyannote]"
```

And a Hugging Face token to download the gated pyannote models. See [pyannote.audio](https://github.com/pyannote/pyannote-audio) for setup.
```

- [ ] **Step 3: Verify README renders correctly**

Run:
```bash
python -m markdown README.md > /dev/null && echo "OK"
```

Expected: `OK` (or use `grip README.md` if installed to preview in browser).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/BACKENDS.md
git commit -m "docs: add README with quickstart, architecture, roadmap and BACKENDS.md"
```

---

## Milestones

| Milestone | Tasks | Deliverable |
|---|---|---|
| **M1 – Skeleton** | 1–4 | Installable package, mock transcription, diarization, CI passing |
| **M2 – Intelligence** | 5–6 | Hebrew/English extractor, ProtocolGenerator |
| **M3 – Outputs + CLI** | 7–8 | All output formats, full `meeting-protocol run` pipeline |
| **M4 – Polish** | 9–11 | Fixtures, lint/type clean, README, BACKENDS docs |
| **M5 – Real backends** | whisper.cpp production use | End-to-end on real audio from Mac mini |

---

## Hebrew + English Transcription Quality Notes

- **Model choice is the single biggest lever.** `large-v3-turbo` gives the best quality/speed tradeoff.
- **Language hint matters.** `--language he` biases the model toward Hebrew; mixed code-switching is handled well by large models.
- **Punctuation in Hebrew** is minimal in whisper output. The extractors use token-level patterns, not sentence boundaries, so they tolerate this.
- **Future:** faster-whisper + `beam_size=5` significantly improves Hebrew recall. pyannote diarization works well with whisperX-aligned timestamps.

---

## Self-Review Checklist

- [x] All spec requirements covered: transcription adapters, diarization, protocol generation, CLI, CI, fixtures, Hebrew/English support, Obsidian output
- [x] No placeholders — all steps have actual code
- [x] Type signatures consistent across tasks (e.g., `Transcript.from_dict` used identically in models.py and cli/main.py)
- [x] `ActionStatus.OPEN` default in `ActionItem` consistent across models and writer table
- [x] `conftest.py` fixtures match segment data used in extractor tests
- [x] No heavyweight optional deps pulled into core `dependencies`
- [x] whisper.cpp adapter documented, not required for tests
