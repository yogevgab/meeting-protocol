import json

import pytest

from meeting_protocol.correction.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    chunk_segments,
    parse_response,
)
from meeting_protocol.models import Segment


def _seg(seg_id: int, text: str = "x") -> Segment:
    return Segment(id=seg_id, start=float(seg_id), end=float(seg_id) + 1.0, text=text)


# ── chunking ──────────────────────────────────────────────────────────────────


def test_chunk_returns_non_overlapping_windows() -> None:
    segs = [_seg(i) for i in range(1, 11)]

    chunks = chunk_segments(segs, window=4, overlap=1)

    assert [len(c) for c in chunks] == [4, 4, 2]
    flat = [s.id for c in chunks for s in c]
    assert flat == list(range(1, 11))


def test_chunk_window_larger_than_segments_returns_one_chunk() -> None:
    segs = [_seg(i) for i in range(1, 4)]

    chunks = chunk_segments(segs, window=10, overlap=2)

    assert chunks == [segs]


def test_chunk_empty_input_returns_empty_list() -> None:
    assert chunk_segments([], window=5, overlap=1) == []


def test_chunk_zero_window_raises() -> None:
    with pytest.raises(ValueError):
        chunk_segments([_seg(1)], window=0, overlap=0)


def test_chunk_negative_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_segments([_seg(1)], window=5, overlap=-1)


# ── prompt building ───────────────────────────────────────────────────────────


def test_build_prompt_includes_title_when_provided() -> None:
    prompt = build_user_prompt(
        chunk=[_seg(1, "hello")],
        secondary_texts=[None],
        context_tail=[],
        title="Roadmap Review",
        participants=[],
    )

    assert "Roadmap Review" in prompt


def test_build_prompt_includes_participants_when_provided() -> None:
    prompt = build_user_prompt(
        chunk=[_seg(1, "hello")],
        secondary_texts=[None],
        context_tail=[],
        title="",
        participants=["Yogev", "Tom"],
    )

    assert "Yogev" in prompt
    assert "Tom" in prompt


def test_build_prompt_uses_text_field_when_no_secondary() -> None:
    prompt = build_user_prompt(
        chunk=[_seg(1, "hello")],
        secondary_texts=[None],
        context_tail=[],
        title="",
        participants=[],
    )

    # Should use the simple {id, text} schema, not {id, primary, secondary}.
    assert "\"text\":" in prompt
    assert "\"primary\":" not in prompt


def test_build_prompt_uses_primary_secondary_when_any_secondary_present() -> None:
    prompt = build_user_prompt(
        chunk=[_seg(1, "hello"), _seg(2, "world")],
        secondary_texts=["Hello!", None],
        context_tail=[],
        title="",
        participants=[],
    )

    assert "\"primary\":" in prompt
    assert "\"secondary\":" in prompt
    assert "Hello!" in prompt


def test_build_prompt_includes_context_tail_when_provided() -> None:
    prompt = build_user_prompt(
        chunk=[_seg(2, "next")],
        secondary_texts=[None],
        context_tail=[_seg(1, "previous")],
        title="",
        participants=[],
    )

    assert "previous" in prompt
    assert "Previously corrected" in prompt


def test_build_prompt_omits_context_section_when_no_tail() -> None:
    prompt = build_user_prompt(
        chunk=[_seg(1, "x")],
        secondary_texts=[None],
        context_tail=[],
        title="",
        participants=[],
    )

    assert "Previously corrected" not in prompt


# ── speaker context ───────────────────────────────────────────────────────────


def _seg_with_speaker(seg_id: int, text: str, speaker_id: str | None) -> Segment:
    return Segment(
        id=seg_id, start=float(seg_id), end=float(seg_id) + 1.0,
        text=text, speaker_id=speaker_id,
    )


def test_speaker_field_emitted_when_speaker_id_set() -> None:
    chunk = [_seg_with_speaker(1, "hello", "SPEAKER_00")]
    prompt = build_user_prompt(
        chunk=chunk,
        secondary_texts=[None],
        context_tail=[],
        title="",
        participants=[],
        speaker_names={"SPEAKER_00": "Yogev"},
    )

    assert "\"speaker\":" in prompt
    assert "Yogev" in prompt


def test_speaker_field_uses_id_when_no_name_mapping() -> None:
    chunk = [_seg_with_speaker(1, "hello", "SPEAKER_42")]
    prompt = build_user_prompt(
        chunk=chunk,
        secondary_texts=[None],
        context_tail=[],
        title="",
        participants=[],
        speaker_names={},
    )

    assert "SPEAKER_42" in prompt


def test_speaker_field_omitted_when_speaker_id_none() -> None:
    chunk = [_seg(1, "hello")]
    prompt = build_user_prompt(
        chunk=chunk,
        secondary_texts=[None],
        context_tail=[],
        title="",
        participants=[],
        speaker_names={"SPEAKER_00": "Yogev"},
    )

    assert "\"speaker\":" not in prompt


def test_speaker_field_in_context_tail_when_set() -> None:
    chunk = [_seg(2, "next")]
    tail = [_seg_with_speaker(1, "previous", "SPEAKER_00")]
    prompt = build_user_prompt(
        chunk=chunk,
        secondary_texts=[None],
        context_tail=tail,
        title="",
        participants=[],
        speaker_names={"SPEAKER_00": "Yogev"},
    )

    # Speaker appears in the prev-corrected context block.
    ctx_block = prompt.split("Window to correct")[0]
    assert "Yogev" in ctx_block


def test_system_prompt_documents_speaker_field_as_read_only_context() -> None:
    """If the speaker-field instruction is removed, the LLM may try to alter
    or echo the 'speaker' field in its response. Lock that paragraph in."""
    assert "'speaker'" in SYSTEM_PROMPT
    lower = SYSTEM_PROMPT.lower()
    assert "do not include" in lower or "do not change" in lower or "read-only" in lower


def test_speaker_field_works_with_secondary_schema() -> None:
    chunk = [_seg_with_speaker(1, "primary", "SPEAKER_00")]
    prompt = build_user_prompt(
        chunk=chunk,
        secondary_texts=["secondary"],
        context_tail=[],
        title="",
        participants=[],
        speaker_names={"SPEAKER_00": "Yogev"},
    )

    assert "\"primary\":" in prompt
    assert "\"secondary\":" in prompt
    assert "\"speaker\":" in prompt
    assert "Yogev" in prompt


# ── response parsing & validation ─────────────────────────────────────────────


def test_parse_response_accepts_well_formed_array() -> None:
    chunk = [_seg(1, "old"), _seg(2, "older")]
    body = json.dumps([{"id": 1, "text": "new"}, {"id": 2, "text": "newer"}])

    texts, warning = parse_response(body, chunk)

    assert warning is None
    assert texts == ["new", "newer"]


def test_parse_response_strips_markdown_fences() -> None:
    chunk = [_seg(1, "old")]
    body = '```json\n[{"id": 1, "text": "new"}]\n```'

    texts, warning = parse_response(body, chunk)

    assert warning is None
    assert texts == ["new"]


def test_parse_response_strips_markdown_fences_with_trailing_whitespace() -> None:
    chunk = [_seg(1, "old")]
    body = '```json\n[{"id": 1, "text": "new"}]\n```   \n'

    texts, warning = parse_response(body, chunk)

    assert warning is None
    assert texts == ["new"]


def test_parse_response_falls_back_on_invalid_json() -> None:
    chunk = [_seg(1, "original")]

    texts, warning = parse_response("not json at all", chunk)

    assert texts == ["original"]
    assert warning is not None


def test_parse_response_falls_back_on_id_mismatch() -> None:
    chunk = [_seg(1, "a"), _seg(2, "b")]
    body = json.dumps([{"id": 1, "text": "a2"}, {"id": 99, "text": "b2"}])

    texts, warning = parse_response(body, chunk)

    assert texts == ["a", "b"]
    assert warning is not None


def test_parse_response_falls_back_on_count_mismatch() -> None:
    chunk = [_seg(1, "a"), _seg(2, "b")]
    body = json.dumps([{"id": 1, "text": "a2"}])

    texts, warning = parse_response(body, chunk)

    assert texts == ["a", "b"]
    assert warning is not None


def test_parse_response_falls_back_on_missing_text() -> None:
    chunk = [_seg(1, "a")]
    body = json.dumps([{"id": 1}])

    texts, warning = parse_response(body, chunk)

    assert texts == ["a"]
    assert warning is not None


def test_parse_response_falls_back_when_text_exceeds_runaway_cap() -> None:
    chunk = [_seg(1, "short")]  # original length 5, cap = 5x max(5, 10) = 50
    body = json.dumps([{"id": 1, "text": "x" * 100}])

    texts, warning = parse_response(body, chunk, runaway_factor=5.0)

    assert texts == ["short"]
    assert warning is not None


def test_parse_response_handles_hebrew_round_trip() -> None:
    chunk = [_seg(1, "שלום, זה בדיקה.")]
    body = json.dumps([{"id": 1, "text": "שלום, זו בדיקה."}], ensure_ascii=False)

    texts, warning = parse_response(body, chunk)

    assert warning is None
    assert texts == ["שלום, זו בדיקה."]


def test_parse_response_falls_back_when_response_is_not_array() -> None:
    chunk = [_seg(1, "x")]
    body = json.dumps({"id": 1, "text": "y"})

    texts, warning = parse_response(body, chunk)

    assert texts == ["x"]
    assert warning is not None
