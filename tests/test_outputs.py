from datetime import datetime

from meeting_protocol.models import (
    ActionItem,
    ActionStatus,
    Decision,
    Highlight,
    Language,
    Protocol,
    Segment,
    Speaker,
    Transcript,
)
from meeting_protocol.outputs.markdown import render_actions, render_protocol, render_transcript


def _sample_protocol() -> Protocol:
    return Protocol(
        title="Test Meeting",
        date=datetime(2024, 1, 15),
        participants=["Yogev", "Tom"],
        source_file="meeting.mp4",
        duration=3720.0,
        decisions=[Decision(description="Launch by end of month.")],
        action_items=[ActionItem(description="Write copy.", owner="Tom")],
        open_questions=["Should we go global from day one?"],
        business_ideas=[Highlight(text="Subscription model idea.", highlight_type="idea")],
        content_opportunities=[
            Highlight(text="Post about the journey.", highlight_type="content_opportunity")
        ],
    )


def test_render_protocol_title() -> None:
    assert "# Test Meeting" in render_protocol(_sample_protocol())


def test_render_protocol_decisions() -> None:
    md = render_protocol(_sample_protocol())
    assert "## Decisions" in md
    assert "Launch by end of month." in md


def test_render_protocol_action_items_open() -> None:
    md = render_protocol(_sample_protocol())
    assert "- [ ]" in md and "Write copy." in md


def test_render_protocol_action_items_done() -> None:
    p = Protocol(
        title="T",
        date=datetime(2024, 1, 1),
        participants=[],
        source_file="x.mp4",
        duration=0.0,
        action_items=[ActionItem(description="Done.", status=ActionStatus.DONE)],
    )
    assert "- [x]" in render_protocol(p)


def test_render_transcript_speakers() -> None:
    t = Transcript(
        segments=[
            Segment(
                id=1,
                start=0.0,
                end=5.0,
                text="שלום",
                speaker_id="S1",
                language=Language.HEBREW,
            ),
            Segment(
                id=2,
                start=5.0,
                end=10.0,
                text="Hello",
                speaker_id="S2",
                language=Language.ENGLISH,
            ),
        ],
        duration=10.0,
        source_file="meeting.mp4",
        speakers=[Speaker(id="S1", name="Yogev"), Speaker(id="S2", name="Tom")],
    )
    md = render_transcript(t)
    assert "Yogev" in md and "שלום" in md
    assert "Tom" in md and "Hello" in md


def test_render_transcript_timestamps() -> None:
    t = Transcript(
        segments=[Segment(id=1, start=65.0, end=70.0, text="hi")],
        duration=70.0,
        source_file="x.mp4",
    )
    assert "01:05" in render_transcript(t)


def test_render_actions_empty() -> None:
    p = Protocol(
        title="Empty",
        date=datetime(2024, 1, 1),
        participants=[],
        source_file="x.mp4",
        duration=0.0,
    )
    assert "No action items" in render_actions(p)


def test_render_actions_with_due_date() -> None:
    p = Protocol(
        title="T",
        date=datetime(2024, 1, 1),
        participants=[],
        source_file="x.mp4",
        duration=0.0,
        action_items=[ActionItem(description="Ship it.", due_date="2024-02-01")],
    )
    assert "2024-02-01" in render_actions(p)
