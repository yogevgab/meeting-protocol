from datetime import datetime

from meeting_protocol.models import (
    ActionItem,
    ActionStatus,
    Language,
    Protocol,
    Segment,
    Speaker,
    Transcript,
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
            Segment(
                id=1,
                start=0.0,
                end=5.2,
                text="שלום",
                speaker_id="S1",
                language=Language.HEBREW,
                confidence=0.95,
            ),
            Segment(
                id=2,
                start=5.5,
                end=10.0,
                text="Hello",
                speaker_id="S2",
                language=Language.ENGLISH,
                confidence=0.97,
            ),
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
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 5.0,
                "text": "hi",
                "speaker_id": None,
                "language": None,
                "confidence": None,
            }
        ],
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
