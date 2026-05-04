import json
from collections.abc import Callable
from pathlib import Path

import pytest

from meeting_protocol.models import Transcript
from meeting_protocol.transcription.whisper_cpp import WhisperCppProvider

# Canonical whisper.cpp --output-json format: offsets in milliseconds, text leading-space-trimmed.
_SAMPLE_OUTPUT = json.dumps(
    {
        "transcription": [
            {
                "timestamps": {"from": "00:00:00,000", "to": "00:00:05,520"},
                "offsets": {"from": 0, "to": 5520},
                "text": " Hello, this is a test.",
            },
            {
                "timestamps": {"from": "00:00:05,520", "to": "00:00:12,800"},
                "offsets": {"from": 5520, "to": 12800},
                "text": " שלום, זה בדיקה.",
            },
        ]
    }
)

SubprocessRunner = Callable[[list[str]], str]


def _fixed_runner(output: str) -> SubprocessRunner:
    def runner(cmd: list[str]) -> str:
        return output

    return runner


# ── command building ──────────────────────────────────────────────────────────


def test_build_command_returns_a_list_of_strings(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(""),
    )
    cmd = provider._build_command(tmp_path / "audio.wav")

    assert isinstance(cmd, list)
    assert all(isinstance(arg, str) for arg in cmd)


def test_build_command_includes_model_path(tmp_path: Path) -> None:
    model = tmp_path / "ggml-base.bin"
    provider = WhisperCppProvider(model_path=model, runner=_fixed_runner(""))
    cmd = provider._build_command(tmp_path / "audio.wav")

    assert str(model) in cmd


def test_build_command_includes_audio_path(tmp_path: Path) -> None:
    model = tmp_path / "ggml-base.bin"
    audio = tmp_path / "audio.wav"
    provider = WhisperCppProvider(model_path=model, runner=_fixed_runner(""))
    cmd = provider._build_command(audio)

    assert str(audio) in cmd


def test_build_command_path_with_spaces_is_single_element(tmp_path: Path) -> None:
    """Paths with spaces must appear as one list element, not shell-split."""
    model = tmp_path / "ggml-base.bin"
    audio = tmp_path / "my meeting audio file.wav"
    provider = WhisperCppProvider(model_path=model, runner=_fixed_runner(""))
    cmd = provider._build_command(audio)

    assert str(audio) in cmd
    # No shell metacharacters should be injected anywhere in the command
    assert not any(";" in arg or "|" in arg or "&&" in arg for arg in cmd)


def test_build_command_uses_configured_whisper_cli_binary(tmp_path: Path) -> None:
    model = tmp_path / "ggml-base.bin"
    provider = WhisperCppProvider(
        model_path=model,
        whisper_cli="/usr/local/bin/whisper-cli",
        runner=_fixed_runner(""),
    )
    cmd = provider._build_command(tmp_path / "audio.wav")

    assert cmd[0] == "/usr/local/bin/whisper-cli"


# ── output parsing ────────────────────────────────────────────────────────────


def test_transcribe_returns_transcript(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
    )
    result = provider.transcribe(tmp_path / "audio.wav")

    assert isinstance(result, Transcript)


def test_transcribe_parses_segment_count(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
    )
    result = provider.transcribe(tmp_path / "audio.wav")

    assert len(result.segments) == 2


def test_transcribe_parses_segment_text_stripped(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
    )
    result = provider.transcribe(tmp_path / "audio.wav")

    assert result.segments[0].text == "Hello, this is a test."
    assert result.segments[1].text == "שלום, זה בדיקה."


def test_transcribe_parses_segment_timestamps_from_offsets(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
    )
    result = provider.transcribe(tmp_path / "audio.wav")

    assert result.segments[0].start == pytest.approx(0.0)
    assert result.segments[0].end == pytest.approx(5.52)
    assert result.segments[1].start == pytest.approx(5.52)
    assert result.segments[1].end == pytest.approx(12.8)


def test_transcribe_duration_equals_last_segment_end(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
    )
    result = provider.transcribe(tmp_path / "audio.wav")

    assert result.duration == pytest.approx(12.8)


def test_transcribe_source_file_matches_audio_path(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
    )
    result = provider.transcribe(audio)

    assert result.source_file == str(audio)


def test_transcribe_segment_ids_are_sequential_from_one(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
    )
    result = provider.transcribe(tmp_path / "audio.wav")

    for i, seg in enumerate(result.segments, start=1):
        assert seg.id == i


# ── runner injection ──────────────────────────────────────────────────────────


def test_runner_receives_the_built_command(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    received: list[list[str]] = []

    def capturing_runner(cmd: list[str]) -> str:
        received.append(cmd)
        return _SAMPLE_OUTPUT

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=capturing_runner,
    )
    provider.transcribe(audio)

    assert len(received) == 1
    assert str(audio) in received[0]


def test_runner_is_called_exactly_once_per_transcribe(tmp_path: Path) -> None:
    call_count = [0]

    def counting_runner(cmd: list[str]) -> str:
        call_count[0] += 1
        return _SAMPLE_OUTPUT

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=counting_runner,
    )
    provider.transcribe(tmp_path / "audio.wav")

    assert call_count[0] == 1
