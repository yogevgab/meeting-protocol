import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from meeting_protocol.correction.base import TranscriptCorrector
from meeting_protocol.correction.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    chunk_segments,
    parse_response,
)
from meeting_protocol.models import Segment, Transcript

HttpClient = Callable[[str, dict[str, Any]], str]
WarningHandler = Callable[[str], None]

# Ollama supports a JSON-schema "format" to constrain output. Without this, models
# like qwen3 return a single object instead of the array we ask for in the prompt.
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "text": {"type": "string"},
        },
        "required": ["id", "text"],
    },
}


def _default_client(url: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 — local Ollama only
        body: str = resp.read().decode("utf-8")
    return body


class OllamaCorrector(TranscriptCorrector):
    def __init__(
        self,
        model: str = "gemma3:12b",
        host: str = "http://localhost:11434",
        window: int = 20,
        overlap: int = 4,
        title: str = "",
        participants: list[str] | None = None,
        runaway_factor: float = 5.0,
        http_client: HttpClient | None = None,
        on_warning: WarningHandler | None = None,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._window = window
        self._overlap = overlap
        self._title = title
        self._participants = participants or []
        self._runaway_factor = runaway_factor
        self._client: HttpClient = http_client if http_client is not None else _default_client
        self._on_warning: WarningHandler = (
            on_warning if on_warning is not None else (lambda _msg: None)
        )

    def correct(
        self,
        transcript: Transcript,
        secondary_texts: list[str | None] | None = None,
    ) -> Transcript:
        if not transcript.segments:
            return transcript
        sec: list[str | None]
        if secondary_texts is None:
            sec = [None] * len(transcript.segments)
        else:
            sec = list(secondary_texts)
        if len(sec) != len(transcript.segments):
            raise ValueError(
                f"secondary_texts length ({len(sec)}) must match "
                f"segments ({len(transcript.segments)})"
            )

        chunks = chunk_segments(transcript.segments, self._window, self._overlap)
        sec_chunks: list[list[str | None]] = [
            sec[i : i + self._window] for i in range(0, len(sec), self._window)
        ]
        speaker_names = {sp.id: sp.name for sp in transcript.speakers}

        corrected_segments: list[Segment] = []
        # context_tail draws from the most-recently corrected segments.
        for chunk, sec_slice in zip(chunks, sec_chunks, strict=True):
            tail = corrected_segments[-self._overlap :] if self._overlap > 0 else []
            try:
                texts = self._correct_chunk(chunk, sec_slice, tail, speaker_names)
            except Exception as e:  # noqa: BLE001 — never let LLM/HTTP failures crash the pipeline
                self._on_warning(f"correction request failed, keeping originals: {e}")
                texts = [s.text for s in chunk]
            for seg, new_text in zip(chunk, texts, strict=True):
                corrected_segments.append(replace(seg, text=new_text))

        return replace(transcript, segments=corrected_segments)

    def _correct_chunk(
        self,
        chunk: list[Segment],
        secondary_texts: list[str | None],
        context_tail: list[Segment],
        speaker_names: dict[str, str],
    ) -> list[str]:
        prompt = build_user_prompt(
            chunk=chunk,
            secondary_texts=secondary_texts,
            context_tail=context_tail,
            title=self._title,
            participants=self._participants,
            speaker_names=speaker_names,
        )
        payload = {
            "model": self._model,
            "stream": False,
            "format": _RESPONSE_SCHEMA,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.0,
            },
        }
        url = f"{self._host}/api/chat"
        body = self._client(url, payload)
        try:
            envelope = json.loads(body)
            content = envelope["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self._on_warning(f"could not extract content from Ollama response: {e}")
            return [s.text for s in chunk]
        texts, warning = parse_response(content, chunk, runaway_factor=self._runaway_factor)
        if warning is not None:
            self._on_warning(warning)
        return texts
