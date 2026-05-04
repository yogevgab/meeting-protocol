import json
from collections.abc import Callable
from typing import Any

from meeting_protocol.correction.ollama import OllamaCorrector
from meeting_protocol.models import Segment, Transcript

HttpCall = tuple[str, dict[str, Any]]


def _seg(seg_id: int, text: str = "x") -> Segment:
    return Segment(id=seg_id, start=float(seg_id), end=float(seg_id) + 1.0, text=text)


def _transcript(*segs: Segment) -> Transcript:
    duration = max((s.end for s in segs), default=0.0)
    return Transcript(segments=list(segs), duration=duration, source_file="/tmp/a.wav")


def _ok_envelope(texts_by_id: dict[int, str]) -> str:
    """Build an Ollama /api/chat non-streaming response whose content is JSON."""
    content = json.dumps(
        [{"id": i, "text": t} for i, t in texts_by_id.items()],
        ensure_ascii=False,
    )
    return json.dumps({"message": {"content": content}})


def _scripted_client(
    responses: list[str],
) -> tuple[Callable[[str, dict[str, Any]], str], list[HttpCall]]:
    calls: list[HttpCall] = []
    queue = list(responses)

    def client(url: str, payload: dict[str, Any]) -> str:
        calls.append((url, payload))
        if not queue:
            raise AssertionError("scripted client ran out of responses")
        return queue.pop(0)

    return client, calls


# ── happy path ────────────────────────────────────────────────────────────────


def test_correct_returns_corrected_text() -> None:
    t = _transcript(_seg(1, "helo"), _seg(2, "wrold"))
    client, _ = _scripted_client([_ok_envelope({1: "hello", 2: "world"})])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client)

    result = corrector.correct(t)

    assert [s.text for s in result.segments] == ["hello", "world"]


def test_correct_preserves_segment_ids_and_timestamps() -> None:
    t = _transcript(_seg(1, "a"), _seg(2, "b"))
    client, _ = _scripted_client([_ok_envelope({1: "A", 2: "B"})])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client)

    result = corrector.correct(t)

    assert [(s.id, s.start, s.end) for s in result.segments] == [
        (1, 1.0, 2.0),
        (2, 2.0, 3.0),
    ]


def test_correct_preserves_speaker_id_when_set() -> None:
    seg = Segment(id=1, start=0.0, end=1.0, text="x", speaker_id="S1")
    t = Transcript(segments=[seg], duration=1.0, source_file="/tmp/a.wav")
    client, _ = _scripted_client([_ok_envelope({1: "X"})])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client)

    result = corrector.correct(t)

    assert result.segments[0].speaker_id == "S1"


def test_correct_preserves_duration_and_source_file() -> None:
    t = _transcript(_seg(1, "a"))
    client, _ = _scripted_client([_ok_envelope({1: "A"})])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client)

    result = corrector.correct(t)

    assert result.duration == t.duration
    assert result.source_file == t.source_file


def test_empty_transcript_does_not_call_http() -> None:
    t = _transcript()
    client, calls = _scripted_client([])
    corrector = OllamaCorrector(http_client=client)

    result = corrector.correct(t)

    assert result.segments == []
    assert calls == []


# ── chunking ──────────────────────────────────────────────────────────────────


def test_correct_chunks_long_transcripts() -> None:
    t = _transcript(*(_seg(i) for i in range(1, 6)))
    responses = [
        _ok_envelope({1: "A", 2: "B"}),
        _ok_envelope({3: "C", 4: "D"}),
        _ok_envelope({5: "E"}),
    ]
    client, calls = _scripted_client(responses)
    corrector = OllamaCorrector(window=2, overlap=0, http_client=client)

    result = corrector.correct(t)

    assert [s.text for s in result.segments] == ["A", "B", "C", "D", "E"]
    assert len(calls) == 3


def test_correct_includes_context_tail_in_subsequent_calls() -> None:
    t = _transcript(_seg(1, "first"), _seg(2, "second"), _seg(3, "third"))
    responses = [
        _ok_envelope({1: "FIRST", 2: "SECOND"}),
        _ok_envelope({3: "THIRD"}),
    ]
    client, calls = _scripted_client(responses)
    corrector = OllamaCorrector(window=2, overlap=2, http_client=client)

    corrector.correct(t)

    # The 2nd call's user prompt should reference the corrected tail (FIRST/SECOND).
    second_user_msg = calls[1][1]["messages"][1]["content"]
    assert "FIRST" in second_user_msg
    assert "SECOND" in second_user_msg


# ── secondary hints ───────────────────────────────────────────────────────────


def test_correct_with_secondary_texts_renders_primary_secondary_schema() -> None:
    t = _transcript(_seg(1, "primary"))
    client, calls = _scripted_client([_ok_envelope({1: "merged"})])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client)

    corrector.correct(t, secondary_texts=["secondary"])

    user_msg = calls[0][1]["messages"][1]["content"]
    assert "primary" in user_msg.lower()
    assert "secondary" in user_msg.lower()


# ── fallback / robustness ─────────────────────────────────────────────────────


def test_invalid_json_response_falls_back_to_originals() -> None:
    t = _transcript(_seg(1, "keep"))
    bad_envelope = json.dumps({"message": {"content": "not a json array"}})
    client, _ = _scripted_client([bad_envelope])
    warnings: list[str] = []
    corrector = OllamaCorrector(
        window=10, overlap=0, http_client=client, on_warning=warnings.append
    )

    result = corrector.correct(t)

    assert [s.text for s in result.segments] == ["keep"]
    assert warnings, "expected a warning to be emitted"


def test_id_mismatch_falls_back_to_originals_for_that_chunk() -> None:
    t = _transcript(_seg(1, "a"), _seg(2, "b"))
    bad = json.dumps(
        {
            "message": {
                "content": json.dumps([{"id": 99, "text": "wrong"}, {"id": 2, "text": "B"}]),
            }
        }
    )
    client, _ = _scripted_client([bad])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client)

    result = corrector.correct(t)

    assert [s.text for s in result.segments] == ["a", "b"]


def test_runaway_length_falls_back_to_originals() -> None:
    t = _transcript(_seg(1, "short"))
    runaway = json.dumps(
        {
            "message": {
                "content": json.dumps([{"id": 1, "text": "x" * 1000}]),
            }
        }
    )
    client, _ = _scripted_client([runaway])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client, runaway_factor=5.0)

    result = corrector.correct(t)

    assert result.segments[0].text == "short"


def test_http_client_exception_keeps_originals_and_warns() -> None:
    t = _transcript(_seg(1, "keep"), _seg(2, "also keep"))

    def failing_client(url: str, payload: dict[str, Any]) -> str:
        raise ConnectionError("ollama not running")

    warnings: list[str] = []
    corrector = OllamaCorrector(
        window=10, overlap=0, http_client=failing_client, on_warning=warnings.append
    )

    result = corrector.correct(t)

    assert [s.text for s in result.segments] == ["keep", "also keep"]
    assert any("ollama not running" in w for w in warnings)


def test_secondary_texts_length_mismatch_raises() -> None:
    t = _transcript(_seg(1, "a"), _seg(2, "b"))
    client, _ = _scripted_client([])
    corrector = OllamaCorrector(http_client=client)

    try:
        corrector.correct(t, secondary_texts=["only one"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ── HTTP wiring ───────────────────────────────────────────────────────────────


def test_request_uses_chat_endpoint_with_configured_host() -> None:
    t = _transcript(_seg(1, "x"))
    client, calls = _scripted_client([_ok_envelope({1: "X"})])
    corrector = OllamaCorrector(host="http://example.local:9999", http_client=client)

    corrector.correct(t)

    assert calls[0][0] == "http://example.local:9999/api/chat"


def test_request_uses_configured_model() -> None:
    t = _transcript(_seg(1, "x"))
    client, calls = _scripted_client([_ok_envelope({1: "X"})])
    corrector = OllamaCorrector(model="my-model:latest", http_client=client)

    corrector.correct(t)

    assert calls[0][1]["model"] == "my-model:latest"


def test_request_disables_streaming_and_forces_array_schema() -> None:
    t = _transcript(_seg(1, "x"))
    client, calls = _scripted_client([_ok_envelope({1: "X"})])
    corrector = OllamaCorrector(http_client=client)

    corrector.correct(t)

    assert calls[0][1]["stream"] is False
    fmt = calls[0][1]["format"]
    assert isinstance(fmt, dict)
    assert fmt["type"] == "array"
    assert fmt["items"]["required"] == ["id", "text"]


def test_hebrew_round_trip_preserves_unicode() -> None:
    t = _transcript(_seg(1, "שלום עולם"))
    response = json.dumps(
        {"message": {"content": json.dumps([{"id": 1, "text": "שלום, עולם"}], ensure_ascii=False)}}
    )
    client, _ = _scripted_client([response])
    corrector = OllamaCorrector(window=10, overlap=0, http_client=client)

    result = corrector.correct(t)

    assert result.segments[0].text == "שלום, עולם"
