"""CLI-level tests for automatic M4A/MP3/AAC normalization.

These tests verify that the end-to-end `transcribe` command accepts compressed
audio formats by monkeypatching the normalizer in `whisper_cpp` and the
subprocess runner, so ffmpeg and whisper-cli are never actually invoked.
"""

import contextlib
import json
from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meeting_protocol.cli.main import app
from meeting_protocol.diarization.base import Diarizer, SpeakerInterval

runner = CliRunner()

_SAMPLE_WHISPER_JSON = json.dumps({
    "transcription": [
        {
            "timestamps": {"from": "00:00:00,000", "to": "00:00:05,000"},
            "offsets": {"from": 0, "to": 5000},
            "text": " Hello world.",
        }
    ]
})


def _audio(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"fake compressed audio")
    return p


class _StubDiarizer(Diarizer):
    def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
        return [SpeakerInterval(start=0.0, end=10.0, label="SPEAKER_00")]


def _make_normalizer(tmp_path: Path, record: list[Path] | None = None):
    """Return a context-manager normalizer that records inputs and yields WAV paths."""
    norm_dir = tmp_path / "_norm"
    norm_dir.mkdir(exist_ok=True)

    @contextlib.contextmanager
    def normalizer(path: Path) -> Generator[Path, None, None]:
        if record is not None:
            record.append(path)
        if path.suffix.lower() != ".wav":
            wav = norm_dir / (path.stem + ".wav")
            wav.write_bytes(b"fake normalized wav")
            yield wav
        else:
            yield path

    return normalizer


def _patch_whisper_cpp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record: list[Path] | None = None,
) -> None:
    """Monkeypatch normalizer and runner in the whisper_cpp module."""
    monkeypatch.setattr(
        "meeting_protocol.transcription.whisper_cpp.normalized_wav",
        _make_normalizer(tmp_path, record),
    )
    monkeypatch.setattr(
        "meeting_protocol.transcription.whisper_cpp._default_runner",
        lambda _cmd: _SAMPLE_WHISPER_JSON,
    )


# ── single M4A file ───────────────────────────────────────────────────────────


def test_single_m4a_whisper_cpp_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe with a single M4A file and whisper-cpp must succeed."""
    _patch_whisper_cpp(monkeypatch, tmp_path)
    m4a = _audio(tmp_path, "voice.m4a")

    result = runner.invoke(app, [
        "transcribe", str(m4a),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    assert result.exit_code == 0, result.output


def test_single_m4a_whisper_cpp_creates_transcript_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcript.json must be written for a single M4A file."""
    _patch_whisper_cpp(monkeypatch, tmp_path)
    m4a = _audio(tmp_path, "voice.m4a")
    out = tmp_path / "out"

    runner.invoke(app, [
        "transcribe", str(m4a),
        "--out", str(out),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    assert (out / "transcript.json").exists()


def test_single_m4a_normalizer_is_called_with_m4a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The normalizer must be called with the original M4A path."""
    normalized: list[Path] = []
    _patch_whisper_cpp(monkeypatch, tmp_path, record=normalized)
    m4a = _audio(tmp_path, "voice.m4a")

    runner.invoke(app, [
        "transcribe", str(m4a),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    assert m4a in normalized


# ── single WAV file (regression) ──────────────────────────────────────────────


def test_single_wav_whisper_cpp_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing WAV behavior must be preserved after the M4A normalization change."""
    _patch_whisper_cpp(monkeypatch, tmp_path)
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF fake wav")

    result = runner.invoke(app, [
        "transcribe", str(wav),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    assert result.exit_code == 0, result.output


# ── multiple M4A files ────────────────────────────────────────────────────────


def test_multi_m4a_whisper_cpp_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe with multiple M4A files must succeed."""
    _patch_whisper_cpp(monkeypatch, tmp_path)
    m4a1 = _audio(tmp_path, "clip1.m4a")
    m4a2 = _audio(tmp_path, "clip2.m4a")

    result = runner.invoke(app, [
        "transcribe", str(m4a1), str(m4a2),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    assert result.exit_code == 0, result.output


def test_multi_m4a_normalizer_called_for_each_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normalizer must be called once per M4A file."""
    normalized: list[Path] = []
    _patch_whisper_cpp(monkeypatch, tmp_path, record=normalized)
    m4a1 = _audio(tmp_path, "clip1.m4a")
    m4a2 = _audio(tmp_path, "clip2.m4a")

    runner.invoke(app, [
        "transcribe", str(m4a1), str(m4a2),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    assert m4a1 in normalized
    assert m4a2 in normalized


# ── multiple M4A + --diarize ──────────────────────────────────────────────────


def test_multi_m4a_diarize_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple M4A files with --diarize must succeed end-to-end."""
    _patch_whisper_cpp(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _StubDiarizer(),
    )
    monkeypatch.setattr(
        "meeting_protocol.cli.main.concat_audio_files",
        lambda paths, out, **kw: out.write_bytes(b"fake wav"),
    )
    m4a1 = _audio(tmp_path, "clip1.m4a")
    m4a2 = _audio(tmp_path, "clip2.m4a")

    result = runner.invoke(app, [
        "transcribe", str(m4a1), str(m4a2),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
        "--diarize",
    ])
    assert result.exit_code == 0, result.output


def test_multi_m4a_diarize_creates_transcript_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcript.json must be written for multi-file M4A + diarize."""
    _patch_whisper_cpp(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _StubDiarizer(),
    )
    monkeypatch.setattr(
        "meeting_protocol.cli.main.concat_audio_files",
        lambda paths, out, **kw: out.write_bytes(b"fake wav"),
    )
    m4a1 = _audio(tmp_path, "clip1.m4a")
    m4a2 = _audio(tmp_path, "clip2.m4a")
    out = tmp_path / "out"

    runner.invoke(app, [
        "transcribe", str(m4a1), str(m4a2),
        "--out", str(out),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
        "--diarize",
    ])
    assert (out / "transcript.json").exists()


def test_multi_m4a_diarize_concat_receives_original_m4a_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """concat_audio_files must receive the original M4A paths, not normalized WAVs.

    ffmpeg inside concat handles compressed inputs; the CLI must not pre-convert
    before calling concat.
    """
    _patch_whisper_cpp(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _StubDiarizer(),
    )
    concat_inputs: list[list[Path]] = []

    def _capture_concat(paths: list[Path], output: Path, **kw: object) -> None:
        concat_inputs.append(list(paths))
        output.write_bytes(b"fake wav")

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _capture_concat)

    m4a1 = _audio(tmp_path, "clip1.m4a")
    m4a2 = _audio(tmp_path, "clip2.m4a")

    runner.invoke(app, [
        "transcribe", str(m4a1), str(m4a2),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
        "--diarize",
    ])
    assert len(concat_inputs) == 1
    assert concat_inputs[0] == [m4a1, m4a2]


def test_multi_m4a_diarize_each_file_normalized_for_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each M4A file must be individually normalized before transcription."""
    normalized: list[Path] = []
    _patch_whisper_cpp(monkeypatch, tmp_path, record=normalized)
    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _StubDiarizer(),
    )
    monkeypatch.setattr(
        "meeting_protocol.cli.main.concat_audio_files",
        lambda paths, out, **kw: out.write_bytes(b"fake wav"),
    )
    m4a1 = _audio(tmp_path, "clip1.m4a")
    m4a2 = _audio(tmp_path, "clip2.m4a")

    runner.invoke(app, [
        "transcribe", str(m4a1), str(m4a2),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
        "--diarize",
    ])
    assert m4a1 in normalized, "clip1.m4a was not normalized for transcription"
    assert m4a2 in normalized, "clip2.m4a was not normalized for transcription"


# ── normalization failure error handling ──────────────────────────────────────


def test_normalization_failure_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If normalization fails (e.g. ffmpeg not found), CLI must exit non-zero."""

    @contextlib.contextmanager
    def failing_normalizer(path: Path) -> Generator[Path, None, None]:
        raise RuntimeError("ffmpeg not found on PATH")
        yield Path("/unreachable")

    monkeypatch.setattr(
        "meeting_protocol.transcription.whisper_cpp.normalized_wav",
        failing_normalizer,
    )
    monkeypatch.setattr(
        "meeting_protocol.transcription.whisper_cpp._default_runner",
        lambda _cmd: _SAMPLE_WHISPER_JSON,
    )
    m4a = _audio(tmp_path, "voice.m4a")

    result = runner.invoke(app, [
        "transcribe", str(m4a),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    assert result.exit_code != 0


def test_normalization_failure_error_message_is_informative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normalization failure message must mention normalization or ffmpeg."""

    @contextlib.contextmanager
    def failing_normalizer(path: Path) -> Generator[Path, None, None]:
        raise RuntimeError("ffmpeg not found on PATH")
        yield Path("/unreachable")

    monkeypatch.setattr(
        "meeting_protocol.transcription.whisper_cpp.normalized_wav",
        failing_normalizer,
    )
    monkeypatch.setattr(
        "meeting_protocol.transcription.whisper_cpp._default_runner",
        lambda _cmd: _SAMPLE_WHISPER_JSON,
    )
    m4a = _audio(tmp_path, "voice.m4a")

    result = runner.invoke(app, [
        "transcribe", str(m4a),
        "--out", str(tmp_path / "out"),
        "--provider", "whisper-cpp",
        "--model", str(tmp_path / "model.bin"),
    ])
    combined_output = result.output + (str(result.exception) if result.exception else "")
    assert (
        "ffmpeg" in combined_output.lower()
        or "normaliz" in combined_output.lower()
        or "audio" in combined_output.lower()
    ), f"Expected informative error, got: {combined_output!r}"
