from pathlib import Path

from typer.testing import CliRunner

from meeting_protocol.cli.main import app
from meeting_protocol.io import load_transcript
from meeting_protocol.models import Transcript

runner = CliRunner()


def _audio(tmp_path: Path) -> Path:
    f = tmp_path / "audio.wav"
    f.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")  # minimal fake WAV header
    return f


# ── output files ──────────────────────────────────────────────────────────────


def test_transcribe_creates_transcript_json(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    assert result.exit_code == 0, result.output
    assert (out / "transcript.json").exists()


def test_transcribe_creates_transcript_md(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    assert result.exit_code == 0, result.output
    assert (out / "transcript.md").exists()


def test_transcribe_creates_protocol_md(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    assert result.exit_code == 0, result.output
    assert (out / "protocol.md").exists()


def test_transcribe_creates_actions_md(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    assert result.exit_code == 0, result.output
    assert (out / "actions.md").exists()


# ── transcript.json content ───────────────────────────────────────────────────


def test_transcript_json_deserialises_to_transcript(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(audio), "--out", str(out), "--provider", "mock"],
    )

    transcript = load_transcript(out / "transcript.json")
    assert isinstance(transcript, Transcript)


def test_transcript_json_source_file_matches_audio_argument(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(audio), "--out", str(out), "--provider", "mock"],
    )

    transcript = load_transcript(out / "transcript.json")
    assert transcript.source_file == str(audio)


def test_transcript_json_has_segments(tmp_path: Path) -> None:
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    transcript = load_transcript(out / "transcript.json")
    assert len(transcript.segments) > 0


# ── protocol.md content ───────────────────────────────────────────────────────


def test_protocol_md_contains_title(tmp_path: Path) -> None:
    out = tmp_path / "out"
    runner.invoke(
        app,
        [
            "transcribe",
            str(_audio(tmp_path)),
            "--out", str(out),
            "--title", "Roadmap Review",
            "--provider", "mock",
        ],
    )

    assert "# Roadmap Review" in (out / "protocol.md").read_text(encoding="utf-8")


def test_protocol_md_contains_participants(tmp_path: Path) -> None:
    out = tmp_path / "out"
    runner.invoke(
        app,
        [
            "transcribe",
            str(_audio(tmp_path)),
            "--out", str(out),
            "--participants", "Yogev,Tom",
            "--provider", "mock",
        ],
    )

    content = (out / "protocol.md").read_text(encoding="utf-8")
    assert "Yogev" in content
    assert "Tom" in content


# ── transcript.md content ─────────────────────────────────────────────────────


def test_transcript_md_contains_timestamp_markers(tmp_path: Path) -> None:
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    # render_transcript produces "[MM:SS → MM:SS]" markers for every segment
    content = (out / "transcript.md").read_text(encoding="utf-8")
    assert "[" in content and "→" in content


# ── actions.md content ────────────────────────────────────────────────────────


def test_actions_md_is_non_empty(tmp_path: Path) -> None:
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    assert (out / "actions.md").read_text(encoding="utf-8").strip()


# ── error handling ────────────────────────────────────────────────────────────


def test_transcribe_exits_nonzero_when_audio_file_missing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(tmp_path / "nonexistent.wav"),
            "--out", str(tmp_path / "out"),
            "--provider", "mock",
        ],
    )

    assert result.exit_code != 0
