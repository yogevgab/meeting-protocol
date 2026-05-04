"""Integration tests for multiple-audio-clip support in the transcribe command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from meeting_protocol.cli.main import app
from meeting_protocol.io import load_transcript
from meeting_protocol.models import Segment, Transcript

runner = CliRunner()


def _audio(tmp_path: Path, name: str = "audio.wav") -> Path:
    f = tmp_path / name
    f.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    return f


def _mock_class(transcripts_by_name: dict[str, Transcript]) -> type:
    """Return a MockTranscriptionProvider class whose transcribe() returns
    per-filename transcripts (looked up by audio_path.name)."""

    class _PerClipMock:
        def transcribe(self, audio_path: Path) -> Transcript:
            return transcripts_by_name[audio_path.name]

    return _PerClipMock


# ── CLI accepts multiple audio paths ─────────────────────────────────────────


def test_transcribe_accepts_two_audio_paths(tmp_path: Path) -> None:
    """CLI must accept two audio paths and exit 0."""
    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock"],
    )
    assert result.exit_code == 0, result.output


def test_transcribe_accepts_three_audio_paths(tmp_path: Path) -> None:
    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    c3 = _audio(tmp_path, "c3.wav")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), str(c3), "--out", str(out), "--provider", "mock"],
    )
    assert result.exit_code == 0, result.output


def test_multiple_clips_create_all_output_files(tmp_path: Path) -> None:
    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock"],
    )
    assert (out / "transcript.json").exists()
    assert (out / "protocol.md").exists()
    assert (out / "transcript.md").exists()
    assert (out / "actions.md").exists()


# ── missing path errors clearly ──────────────────────────────────────────────


def test_exits_nonzero_when_second_clip_missing(tmp_path: Path) -> None:
    """If the second path does not exist, must exit non-zero and name the bad file."""
    c1 = _audio(tmp_path, "c1.wav")
    missing = tmp_path / "missing.wav"
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(c1), str(missing), "--out", str(out), "--provider", "mock"],
    )
    assert result.exit_code != 0
    assert "missing.wav" in result.output


def test_exits_nonzero_when_first_clip_missing(tmp_path: Path) -> None:
    missing = tmp_path / "gone.wav"
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(missing), str(c2), "--out", str(out), "--provider", "mock"],
    )
    assert result.exit_code != 0
    assert "gone.wav" in result.output


# ── merged transcript preserves order and offsets timestamps ─────────────────


def test_second_clip_segments_are_offset_by_first_clip_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Segments from clip2 (starting at 0.0 in its own timebase) must be
    offset by clip1's last segment end (20.0) in the merged transcript."""

    clip1_transcript = Transcript(
        segments=[
            Segment(id=1, start=0.0, end=5.0, text="Hello"),
            Segment(id=2, start=5.5, end=10.0, text="World"),
            Segment(id=3, start=10.5, end=20.0, text="Done"),
        ],
        duration=20.0,
        source_file=str(tmp_path / "c1.wav"),
    )
    clip2_transcript = Transcript(
        segments=[
            Segment(id=1, start=0.0, end=4.0, text="New clip"),
            Segment(id=2, start=4.5, end=8.0, text="More content"),
        ],
        duration=8.0,
        source_file=str(tmp_path / "c2.wav"),
    )

    monkeypatch.setattr(
        "meeting_protocol.cli.main.MockTranscriptionProvider",
        _mock_class({"c1.wav": clip1_transcript, "c2.wav": clip2_transcript}),
    )

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock"],
    )

    transcript = load_transcript(out / "transcript.json")
    assert len(transcript.segments) == 5

    # clip1 segments at original positions
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[2].end == 20.0

    # clip2 first segment: 0.0 + 20.0 = 20.0
    assert transcript.segments[3].start == 20.0
    assert transcript.segments[3].end == 24.0  # 4.0 + 20.0

    # clip2 second segment: 4.5 + 20.0 = 24.5
    assert transcript.segments[4].start == 24.5
    assert transcript.segments[4].end == 28.0  # 8.0 + 20.0


def test_merged_transcript_segments_in_clip_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All segments from clip1 must precede all segments from clip2."""
    clip1_transcript = Transcript(
        segments=[
            Segment(id=1, start=0.0, end=5.0, text="Clip1 seg1"),
            Segment(id=2, start=5.0, end=10.0, text="Clip1 seg2"),
        ],
        duration=10.0,
        source_file=str(tmp_path / "c1.wav"),
    )
    clip2_transcript = Transcript(
        segments=[
            Segment(id=1, start=0.0, end=5.0, text="Clip2 seg1"),
            Segment(id=2, start=5.0, end=10.0, text="Clip2 seg2"),
        ],
        duration=10.0,
        source_file=str(tmp_path / "c2.wav"),
    )

    monkeypatch.setattr(
        "meeting_protocol.cli.main.MockTranscriptionProvider",
        _mock_class({"c1.wav": clip1_transcript, "c2.wav": clip2_transcript}),
    )

    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock"],
    )

    transcript = load_transcript(out / "transcript.json")
    texts = [s.text for s in transcript.segments]
    assert texts == ["Clip1 seg1", "Clip1 seg2", "Clip2 seg1", "Clip2 seg2"]


# ── raw outputs remain one meeting ───────────────────────────────────────────


def test_three_clips_produce_single_transcript_json(tmp_path: Path) -> None:
    """Three clips must produce one transcript.json (not per-clip files)."""
    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    c3 = _audio(tmp_path, "c3.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), str(c3), "--out", str(out), "--provider", "mock"],
    )
    json_files = list(out.glob("transcript*.json"))
    assert len(json_files) == 1
    assert (out / "transcript.json").exists()


def test_multiple_clips_merged_transcript_is_valid_transcript(tmp_path: Path) -> None:
    c1 = _audio(tmp_path, "c1.wav")
    c2 = _audio(tmp_path, "c2.wav")
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(out), "--provider", "mock"],
    )
    transcript = load_transcript(out / "transcript.json")
    assert isinstance(transcript, Transcript)
    assert len(transcript.segments) > 0


def test_multiple_clips_merged_transcript_has_more_segments_than_single(
    tmp_path: Path,
) -> None:
    """Merged transcript from two clips (same mock provider) must have 2× segments."""
    single_out = tmp_path / "single"
    c1 = _audio(tmp_path, "single.wav")
    runner.invoke(
        app,
        ["transcribe", str(c1), "--out", str(single_out), "--provider", "mock"],
    )
    single_count = len(load_transcript(single_out / "transcript.json").segments)

    multi_out = tmp_path / "multi"
    c2 = _audio(tmp_path, "c2.wav")
    runner.invoke(
        app,
        ["transcribe", str(c1), str(c2), "--out", str(multi_out), "--provider", "mock"],
    )
    multi_count = len(load_transcript(multi_out / "transcript.json").segments)

    assert multi_count == single_count * 2


# ── single-file backward compatibility ───────────────────────────────────────


def test_single_clip_still_exits_zero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(tmp_path / "out"), "--provider", "mock"],
    )
    assert result.exit_code == 0, result.output


def test_single_clip_transcript_json_source_file_unchanged(tmp_path: Path) -> None:
    """With one path, source_file must still equal the audio path (backward compat)."""
    audio = _audio(tmp_path)
    out = tmp_path / "out"
    runner.invoke(
        app,
        ["transcribe", str(audio), "--out", str(out), "--provider", "mock"],
    )
    transcript = load_transcript(out / "transcript.json")
    assert transcript.source_file == str(audio)
