from meeting_protocol.models import Language, Segment, Transcript
from meeting_protocol.protocol.generator import generate_protocol


def _make_transcript(segments_text: list[tuple[str, str]]) -> Transcript:
    segments = [
        Segment(
            id=i + 1,
            start=float(i * 5),
            end=float((i + 1) * 5),
            text=text,
            language=Language(lang),
        )
        for i, (text, lang) in enumerate(segments_text)
    ]
    return Transcript(
        segments=segments,
        duration=float(len(segments) * 5),
        source_file="test.mp4",
    )


def test_extract_english_decision() -> None:
    t = _make_transcript([("We decided to launch next week.", "en")])
    p = generate_protocol(t, "Test", ["Yogev", "Tom"])
    assert len(p.decisions) == 1


def test_extract_hebrew_decision() -> None:
    t = _make_transcript([("הוחלט שנעלה בסוף החודש.", "he")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.decisions) == 1


def test_extract_english_action() -> None:
    t = _make_transcript([("I'll follow up with the designer.", "en")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.action_items) == 1


def test_extract_hebrew_action() -> None:
    t = _make_transcript([("אנחנו צריכים לסיים את הדף.", "he")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.action_items) == 1


def test_extract_business_idea_en() -> None:
    t = _make_transcript([("Business idea: we could offer a free tier.", "en")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.business_ideas) == 1
    assert p.business_ideas[0].highlight_type == "idea"


def test_extract_hebrew_business_idea() -> None:
    t = _make_transcript([("יש לנו רעיון עסקי לגבי חבילות פרמיום.", "he")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.business_ideas) == 1


def test_extract_content_opportunity_en() -> None:
    t = _make_transcript([("We could write a post about building local AI tools.", "en")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.content_opportunities) == 1


def test_extract_content_opportunity_he() -> None:
    t = _make_transcript([("פוסט על הכלים שאנחנו משתמשים בהם.", "he")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.content_opportunities) == 1


def test_extract_open_question() -> None:
    t = _make_transcript([("Open question: should we go global from day one?", "en")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert len(p.open_questions) == 1


def test_no_false_positives_on_plain_text() -> None:
    t = _make_transcript([("The weather is nice today.", "en"), ("נתראה מחר.", "he")])
    p = generate_protocol(t, "Test", ["Yogev"])
    assert p.decisions == []
    assert p.action_items == []
    assert p.business_ideas == []
