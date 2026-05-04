"""Integration tests: multi-file audio inputs with --diarize."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from meeting_protocol.cli.main import app
from meeting_protocol.diarization.base import Diarizer, SpeakerInterval
from meeting_protocol.io import load_transcript

runner = CliRunner()


def _audio(tmp_path: Path, name: str) -> Path:
    f = tmp_path / name
    f.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    return f


class _StubDiarizer(Diarizer):
    def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
        return [
            SpeakerInterval(start=0.0, end=50.0, label="SPEAKER_00"),
            SpeakerInterval(start=50.0, end=200.0, label="SPEAKER_01"),
        ]


def _stub_diarizer_builder(**_kw: object) -> Diarizer:
    return _StubDiarizer()


def _noop_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
    """Fake concat that just creates the output file without calling ffmpeg."""
    output.write_bytes(b"fake_audio")


# ── main feature: multiple files + --diarize ─────────────────────────────────


def test_multi_file_diarize_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe with 2 audio files and --diarize must succeed (exit 0)."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)
    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _noop_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    result = runner.invoke(
        app,
        [
            "transcribe", str(c1), str(c2),
            "--out", str(tmp_path / "out"),
            "--provider", "mock", "--diarize",
        ],
    )
    assert result.exit_code == 0, result.output


def test_multi_file_diarize_creates_transcript_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)
    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _noop_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock", "--diarize"],
    )
    assert (out / "transcript.json").exists()


def test_multi_file_diarize_writes_raw_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--diarize is a post-stage: transcript.raw.json must be written."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)
    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _noop_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock", "--diarize"],
    )
    assert (out / "transcript.raw.json").exists()


def test_multi_file_diarize_assigns_speaker_ids_to_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After diarization all merged segments must have a speaker_id populated."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)
    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _noop_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock", "--diarize"],
    )
    transcript = load_transcript(out / "transcript.json")
    assert len(transcript.segments) > 0
    speaker_ids = {s.speaker_id for s in transcript.segments}
    # At least some segments must have received a speaker ID
    assert speaker_ids - {None}, f"No speaker IDs assigned; got: {speaker_ids}"


def test_multi_file_diarize_populates_speakers_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcript.speakers must be non-empty after multi-file diarization."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)
    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _noop_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock", "--diarize"],
    )
    transcript = load_transcript(out / "transcript.json")
    assert len(transcript.speakers) >= 1


# ── concat is called correctly ────────────────────────────────────────────────


def test_multi_file_diarize_concat_called_with_inputs_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """concat_audio_files must receive exactly the CLI audio paths in argument order."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)

    concat_calls: list[list[Path]] = []

    def _capture_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
        concat_calls.append(list(paths))
        output.write_bytes(b"fake")

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _capture_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    runner.invoke(
        app,
        [
            "transcribe", str(c1), str(c2),
            "--out", str(tmp_path / "out"),
            "--provider", "mock", "--diarize",
        ],
    )
    assert len(concat_calls) == 1
    assert concat_calls[0] == [c1, c2]


def test_multi_file_diarize_concat_called_once_for_three_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With three audio files, concat must be called exactly once with all three paths."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)

    concat_calls: list[list[Path]] = []

    def _capture_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
        concat_calls.append(list(paths))
        output.write_bytes(b"fake")

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _capture_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    c3 = _audio(tmp_path, "c3.wav")
    runner.invoke(
        app,
        [
            "transcribe", str(c1), str(c2), str(c3),
            "--out", str(tmp_path / "out"),
            "--provider", "mock", "--diarize",
        ],
    )
    assert len(concat_calls) == 1
    assert concat_calls[0] == [c1, c2, c3]


def test_multi_file_diarize_diarizer_receives_concatenated_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diarizer must be called with the concat output path, not the original inputs."""
    diarizer_paths: list[Path] = []

    class _CapturingDiarizer(Diarizer):
        def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
            diarizer_paths.append(audio_path)
            return [SpeakerInterval(start=0.0, end=100.0, label="SPEAKER_00")]

    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _CapturingDiarizer(),
    )

    concat_outputs: list[Path] = []

    def _capture_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
        concat_outputs.append(output)
        output.write_bytes(b"fake")

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _capture_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    runner.invoke(
        app,
        [
            "transcribe", str(c1), str(c2),
            "--out", str(tmp_path / "out"),
            "--provider", "mock", "--diarize",
        ],
    )
    assert len(diarizer_paths) == 1, "diarizer was not called"
    assert len(concat_outputs) == 1, "concat was not called"
    assert diarizer_paths[0] == concat_outputs[0], "diarizer did not receive the concat output"
    assert diarizer_paths[0] not in [c1, c2], "diarizer was called with an original input path"


# ── error handling ────────────────────────────────────────────────────────────


def test_multi_file_diarize_concat_error_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If audio concatenation fails, CLI must exit non-zero."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)

    def _fail_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
        raise RuntimeError("ffmpeg not found on PATH")

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _fail_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    result = runner.invoke(
        app,
        [
            "transcribe", str(c1), str(c2),
            "--out", str(tmp_path / "out"),
            "--provider", "mock", "--diarize",
        ],
    )
    assert result.exit_code != 0


def test_multi_file_diarize_concat_error_mentions_concatenation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error message must mention 'concatenation' or 'audio'."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)

    def _fail_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
        raise RuntimeError("ffmpeg not found on PATH")

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _fail_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    result = runner.invoke(
        app,
        [
            "transcribe", str(c1), str(c2),
            "--out", str(tmp_path / "out"),
            "--provider", "mock", "--diarize",
        ],
    )
    assert "concatenat" in result.output.lower() or "audio" in result.output.lower()


# ── speaker-map works with multiple files ─────────────────────────────────────


def test_multi_file_diarize_with_speaker_map_renames_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--speaker-map must rename speakers when diarizing multiple files."""

    class _OneSpeakerDiarizer(Diarizer):
        def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
            return [SpeakerInterval(start=0.0, end=200.0, label="SPEAKER_00")]

    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _OneSpeakerDiarizer(),
    )
    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _noop_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(c1), str(c2),
            "--out", str(out),
            "--provider", "mock", "--diarize",
            "--speaker-map", "SPEAKER_00=Yogev",
        ],
    )
    assert result.exit_code == 0, result.output
    transcript = load_transcript(out / "transcript.json")
    names = {sp.id: sp.name for sp in transcript.speakers}
    assert names.get("SPEAKER_00") == "Yogev"


# ── backward compatibility: single file is unchanged ─────────────────────────


def test_single_file_diarize_does_not_call_concat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-file --diarize must NOT invoke concat_audio_files."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)

    concat_called: list[bool] = []

    def _track_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
        concat_called.append(True)

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _track_concat)

    c1 = _audio(tmp_path, "c1.wav")
    runner.invoke(
        app,
        ["transcribe", str(c1), "--out", str(tmp_path / "out"), "--provider", "mock", "--diarize"],
    )
    assert not concat_called, "concat_audio_files was called for a single-file transcription"


def test_single_file_diarize_still_assigns_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing single-file --diarize behavior must be preserved after this change."""
    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _stub_diarizer_builder)

    c1 = _audio(tmp_path, "c1.wav")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(c1), "--out", str(out), "--provider", "mock", "--diarize"],
    )
    assert result.exit_code == 0, result.output
    transcript = load_transcript(out / "transcript.json")
    speaker_ids = {s.speaker_id for s in transcript.segments}
    assert speaker_ids - {None}, "Single-file diarize no longer assigns speaker IDs"


def test_multi_file_no_diarize_does_not_call_concat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple files without --diarize must NOT invoke concat_audio_files."""
    concat_called: list[bool] = []

    def _track_concat(paths: list[Path], output: Path, **kwargs: object) -> None:
        concat_called.append(True)

    monkeypatch.setattr("meeting_protocol.cli.main.concat_audio_files", _track_concat)

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(tmp_path / "out"), "--provider", "mock"],
    )
    assert not concat_called, "concat_audio_files was called without --diarize"
