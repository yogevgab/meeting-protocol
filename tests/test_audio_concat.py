"""Unit tests for the audio concatenation utility."""

import subprocess
from pathlib import Path

import pytest

from meeting_protocol.audio.concat import concat_audio_files


def _make_wav(path: Path) -> Path:
    path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    return path


def _filelist_path_from_cmd(cmd: list[str]) -> Path | None:
    """Extract the -i <filelist> path from an ffmpeg concat command."""
    try:
        i_idx = cmd.index("-i")
        return Path(cmd[i_idx + 1])
    except (ValueError, IndexError):
        return None


# ── subprocess invocation ─────────────────────────────────────────────────────


def test_concat_calls_subprocess_exactly_once(tmp_path: Path) -> None:
    """concat_audio_files must invoke the subprocess runner exactly once."""
    calls: list[int] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(1)
        # Create the output file the caller expects to exist
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    inputs = [_make_wav(tmp_path / "a.wav"), _make_wav(tmp_path / "b.wav")]
    concat_audio_files(inputs, tmp_path / "out.wav", _run=_fake_run)

    assert len(calls) == 1


def test_concat_output_path_is_last_arg(tmp_path: Path) -> None:
    """The output path must appear as the final argument in the subprocess call."""
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    out = tmp_path / "out.wav"
    inputs = [_make_wav(tmp_path / "a.wav"), _make_wav(tmp_path / "b.wav")]
    concat_audio_files(inputs, out, _run=_fake_run)

    assert captured[0][-1] == str(out)


def test_concat_normalizes_output_for_diarization(tmp_path: Path) -> None:
    """The concat command must produce mono 16 kHz PCM WAV for pyannote."""
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    inputs = [_make_wav(tmp_path / "a.wav"), _make_wav(tmp_path / "b.wav")]
    concat_audio_files(inputs, tmp_path / "out.wav", _run=_fake_run)

    cmd = captured[0]
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    assert "-vn" in cmd


# ── filelist contents ─────────────────────────────────────────────────────────


def test_concat_filelist_contains_both_input_paths(tmp_path: Path) -> None:
    """The filelist passed to ffmpeg must contain both input file paths."""
    filelist_content: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        list_path = _filelist_path_from_cmd(cmd)
        if list_path and list_path.exists():
            filelist_content.append(list_path.read_text(encoding="utf-8"))
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    a = _make_wav(tmp_path / "a.wav")
    b = _make_wav(tmp_path / "b.wav")
    concat_audio_files([a, b], tmp_path / "out.wav", _run=_fake_run)

    assert filelist_content, "filelist was not read inside _run"
    content = filelist_content[0]
    assert str(a.resolve()) in content
    assert str(b.resolve()) in content


def test_concat_filelist_preserves_input_order(tmp_path: Path) -> None:
    """The filelist must list a.wav before b.wav (matches CLI argument order)."""
    filelist_content: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        list_path = _filelist_path_from_cmd(cmd)
        if list_path and list_path.exists():
            filelist_content.append(list_path.read_text(encoding="utf-8"))
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    a = _make_wav(tmp_path / "a.wav")
    b = _make_wav(tmp_path / "b.wav")
    concat_audio_files([a, b], tmp_path / "out.wav", _run=_fake_run)

    content = filelist_content[0]
    assert content.index(str(a.resolve())) < content.index(str(b.resolve()))


# ── error handling ────────────────────────────────────────────────────────────


def test_concat_raises_runtime_error_on_nonzero_exit(tmp_path: Path) -> None:
    """Non-zero ffmpeg exit must raise RuntimeError mentioning 'ffmpeg'."""

    def _fail(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, 1, b"", b"some ffmpeg error")

    inputs = [_make_wav(tmp_path / "a.wav"), _make_wav(tmp_path / "b.wav")]
    with pytest.raises(RuntimeError, match="ffmpeg"):
        concat_audio_files(inputs, tmp_path / "out.wav", _run=_fail)


# ── filelist cleanup ──────────────────────────────────────────────────────────


def test_concat_filelist_is_removed_after_success(tmp_path: Path) -> None:
    """No concat filelist must remain after a successful call."""
    seen_list: list[Path] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        list_path = _filelist_path_from_cmd(cmd)
        if list_path:
            seen_list.append(list_path)
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    inputs = [_make_wav(tmp_path / "a.wav"), _make_wav(tmp_path / "b.wav")]
    concat_audio_files(inputs, tmp_path / "out.wav", _run=_fake_run)

    assert seen_list, "filelist path not captured"
    assert not seen_list[0].exists(), "filelist was not cleaned up after success"


def test_concat_filelist_is_removed_after_failure(tmp_path: Path) -> None:
    """No concat filelist must remain even when ffmpeg fails."""
    seen_list: list[Path] = []

    def _fail(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        list_path = _filelist_path_from_cmd(cmd)
        if list_path:
            seen_list.append(list_path)
        return subprocess.CompletedProcess(cmd, 1, b"", b"error")

    inputs = [_make_wav(tmp_path / "a.wav"), _make_wav(tmp_path / "b.wav")]
    with pytest.raises(RuntimeError):
        concat_audio_files(inputs, tmp_path / "out.wav", _run=_fail)

    assert seen_list, "filelist path not captured"
    assert not seen_list[0].exists(), "filelist was not cleaned up after failure"
