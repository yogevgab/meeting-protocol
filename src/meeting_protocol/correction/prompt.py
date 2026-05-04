import json
from dataclasses import dataclass

from meeting_protocol.models import Segment

SYSTEM_PROMPT = (
    "You are a conservative transcript corrector. The transcript was produced by "
    "automatic speech recognition (ASR) from an audio recording. ASR makes occasional "
    "errors: mishearing similar-sounding words, garbling proper names, breaking word "
    "boundaries, and inserting raw control characters.\n\n"
    "Fix only clear ASR errors. Do NOT paraphrase, summarize, translate, smooth "
    "disfluencies, normalize punctuation style, or change the speaker's wording. "
    "Do NOT add or remove segments. Do NOT reorder segments. When in doubt, keep "
    "the original text.\n\n"
    "When two ASR outputs (primary and secondary) are provided per segment and they "
    "agree, use that. When they disagree, pick whichever fits the surrounding "
    "context. When neither fits, keep the primary.\n\n"
    "If a 'speaker' field is present, treat it as read-only context (it tells you "
    "who is speaking — useful for proper-name disambiguation). Do NOT include "
    "'speaker' in your response and do NOT change it.\n\n"
    "Output ONLY a JSON array, no commentary, no markdown fences. Format:\n"
    '[{"id": <int>, "text": "<corrected text>"}, ...]\n'
    "The same ids as the input window, in the same order, exactly one entry per segment."
)


@dataclass
class Chunk:
    """A non-overlapping window of segments to correct, plus an optional read-only context tail."""

    segments: list[Segment]
    context_tail: list[Segment]  # already-corrected previous segments, prompt-only


def chunk_segments(
    segments: list[Segment],
    window: int,
    overlap: int,
) -> list[list[Segment]]:
    """Split segments into non-overlapping windows of size `window`. The `overlap`
    parameter is consumed at prompt time (as context_tail), not by changing chunk
    boundaries — this keeps each segment owned by exactly one chunk."""
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    return [segments[i : i + window] for i in range(0, len(segments), window)]


def build_user_prompt(
    chunk: list[Segment],
    secondary_texts: list[str | None],
    context_tail: list[Segment],
    title: str,
    participants: list[str],
    speaker_names: dict[str, str] | None = None,
) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"Meeting: {title}")
    if participants:
        parts.append("Participants: " + ", ".join(participants))

    name_for = speaker_names or {}

    def _speaker(seg: Segment) -> str | None:
        if seg.speaker_id is None:
            return None
        return name_for.get(seg.speaker_id, seg.speaker_id)

    if context_tail:
        ctx = []
        for s in context_tail:
            entry: dict[str, object] = {"id": s.id, "text": s.text}
            spk = _speaker(s)
            if spk is not None:
                entry["speaker"] = spk
            ctx.append(entry)
        parts.append(
            "Previously corrected (context only — do NOT include in your response):\n"
            + json.dumps(ctx, ensure_ascii=False)
        )

    has_secondary = any(t is not None for t in secondary_texts)
    window: list[dict[str, object]] = []
    for s, sec in zip(chunk, secondary_texts, strict=True):
        if has_secondary:
            entry = {
                "id": s.id,
                "primary": s.text,
                "secondary": sec if sec is not None else "",
            }
        else:
            entry = {"id": s.id, "text": s.text}
        spk = _speaker(s)
        if spk is not None:
            entry["speaker"] = spk
        window.append(entry)

    parts.append(
        "Window to correct (return one entry per id, same order):\n"
        + json.dumps(window, ensure_ascii=False)
    )
    return "\n\n".join(parts)


def parse_response(
    response: str,
    chunk: list[Segment],
    runaway_factor: float = 5.0,
) -> tuple[list[str], str | None]:
    """Validate the LLM response and return (texts, warning).

    On any validation failure, return the original chunk texts and a warning string.
    Never raises."""
    expected_ids = [s.id for s in chunk]
    fallback = [s.text for s in chunk]

    body = response.strip()
    # Strip common markdown code fences if model ignored format instructions.
    if body.startswith("```"):
        body = body.lstrip("`")
        # remove possible language tag (json\n)
        if "\n" in body:
            first, rest = body.split("\n", 1)
            if first.strip().lower() in {"json", ""}:
                body = rest
        if body.endswith("```"):
            body = body[: -3]
        body = body.strip()

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return fallback, f"response was not valid JSON: {e}"

    if not isinstance(data, list):
        return fallback, "response was not a JSON array"
    if len(data) != len(chunk):
        return fallback, f"response had {len(data)} entries, expected {len(chunk)}"

    texts: list[str] = []
    for i, (item, original) in enumerate(zip(data, chunk, strict=True)):
        if not isinstance(item, dict):
            return fallback, f"entry {i} was not an object"
        if item.get("id") != expected_ids[i]:
            return fallback, f"entry {i} had id {item.get('id')!r}, expected {expected_ids[i]}"
        text = item.get("text")
        if not isinstance(text, str):
            return fallback, f"entry {i} 'text' was not a string"
        cap = max(len(original.text), 10) * runaway_factor
        if len(text) > cap:
            return (
                fallback,
                f"entry {i} text length {len(text)} exceeds runaway cap {cap:.0f}",
            )
        texts.append(text)
    return texts, None
