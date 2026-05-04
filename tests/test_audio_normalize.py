"""Tests for the audio normalization helper (normalize_to_wav context manager)."""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from meeting_protocol.audio.normalize import normalized_wav


def _fake_run(
    returncode: int = 0,
    stderr: bytes = b"",
) -> Callable:
    """Return a fake subprocess.run that writes a placeholder output file."""

    def _run(cmd: list, *, capture_output: bool = False, **kwargs) -> subprocess.CompletedProcess:
        Path(cmd[-1]).write_bytes(b"fake_wav_content")
        return subprocess.CompletedProcess(cmd, returncode, b"", stderr)

    return _run


# ── WAV passthrough ───────────────────────────────────────────────────────────


def test_wav_file_is_yielded_unchanged(tmp_path: Path) -> None:
    """WAV files must pass through without invoking ffmpeg."""
    wav = tmp_path / "recording.wav"
    wav.write_bytes(b"RIFF fake wav")
    ffmpeg_called = False

    def _track(cmd: list, **kwargs) -> subprocess.CompletedProcess:
        nonlocal ffmpeg_called
        ffmpeg_called = True
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(wav, _run=_track) as result:
        assert result == wav
    assert not ffmpeg_called


def test_wav_extension_case_insensitive_passes_through(tmp_path: Path) -> None:
    """.WAV (uppercase) must also pass through without ffmpeg."""
    wav = tmp_path / "RECORDING.WAV"
    wav.write_bytes(b"RIFF fake wav")
    ffmpeg_called = False

    def _track(cmd: list, **kwargs) -> subprocess.CompletedProcess:
        nonlocal ffmpeg_called
        ffmpeg_called = True
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(wav, _run=_track) as result:
        assert result == wav
    assert not ffmpeg_called


# ── M4A / MP3 / AAC trigger conversion ───────────────────────────────────────


def test_m4a_triggers_ffmpeg_conversion(tmp_path: Path) -> None:
    """M4A files must trigger an ffmpeg subprocess call."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")
    ffmpeg_called = False

    def _track(cmd: list, **kwargs) -> subprocess.CompletedProcess:
        nonlocal ffmpeg_called
        ffmpeg_called = True
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(m4a, _run=_track):
        pass
    assert ffmpeg_called


def test_mp3_triggers_ffmpeg_conversion(tmp_path: Path) -> None:
    """MP3 files must trigger ffmpeg conversion."""
    mp3 = tmp_path / "audio.mp3"
    mp3.write_bytes(b"ID3 fake mp3")
    ffmpeg_called = False

    def _track(cmd: list, **kwargs) -> subprocess.CompletedProcess:
        nonlocal ffmpeg_called
        ffmpeg_called = True
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(mp3, _run=_track):
        pass
    assert ffmpeg_called


def test_aac_triggers_ffmpeg_conversion(tmp_path: Path) -> None:
    """AAC files must trigger ffmpeg conversion."""
    aac = tmp_path / "audio.aac"
    aac.write_bytes(b"fake aac")
    ffmpeg_called = False

    def _track(cmd: list, **kwargs) -> subprocess.CompletedProcess:
        nonlocal ffmpeg_called
        ffmpeg_called = True
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(aac, _run=_track):
        pass
    assert ffmpeg_called


# ── ffmpeg argument correctness ───────────────────────────────────────────────


def test_ffmpeg_receives_input_flag_pointing_to_original(tmp_path: Path) -> None:
    """ffmpeg -i argument must be the original compressed file."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")
    captured: list[list[str]] = []

    def _capture(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(m4a, _run=_capture):
        pass

    cmd = captured[0]
    i_idx = cmd.index("-i")
    assert cmd[i_idx + 1] == str(m4a)


def test_ffmpeg_normalizes_to_mono_16khz_pcm(tmp_path: Path) -> None:
    """ffmpeg must produce mono 16 kHz pcm_s16le WAV for whisper-cli compatibility."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")
    captured: list[list[str]] = []

    def _capture(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(m4a, _run=_capture):
        pass

    cmd = captured[0]
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"


def test_ffmpeg_output_is_last_arg(tmp_path: Path) -> None:
    """The output file path must be the last argument in the ffmpeg command."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")
    captured: list[list[str]] = []

    def _capture(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        captured.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    with normalized_wav(m4a, _run=_capture) as wav_path:
        pass

    assert captured[0][-1] == str(wav_path)


# ── yielded path shape ────────────────────────────────────────────────────────


def test_normalized_path_has_wav_suffix(tmp_path: Path) -> None:
    """The yielded path for a non-WAV file must have .wav extension."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")

    with normalized_wav(m4a, _run=_fake_run()) as result:
        assert result.suffix.lower() == ".wav"


def test_normalized_path_preserves_original_stem(tmp_path: Path) -> None:
    """The normalized WAV stem must match the original file stem."""
    m4a = tmp_path / "my_meeting.m4a"
    m4a.write_bytes(b"fake m4a")

    with normalized_wav(m4a, _run=_fake_run()) as result:
        assert result.stem == "my_meeting"


# ── temp-file cleanup ─────────────────────────────────────────────────────────


def test_temp_wav_file_is_deleted_after_context_exits(tmp_path: Path) -> None:
    """The temporary WAV file must not exist after the context manager exits."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")
    captured: list[Path] = []

    with normalized_wav(m4a, _run=_fake_run()) as wav_path:
        captured.append(wav_path)

    assert not captured[0].exists(), "Temp WAV was not cleaned up"


def test_temp_wav_file_exists_inside_context(tmp_path: Path) -> None:
    """The yielded path must exist while inside the context."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")

    with normalized_wav(m4a, _run=_fake_run()) as wav_path:
        assert wav_path.exists(), "Temp WAV does not exist inside context"


# ── error handling ────────────────────────────────────────────────────────────


def test_ffmpeg_nonzero_exit_raises_runtime_error(tmp_path: Path) -> None:
    """Non-zero ffmpeg exit code must raise RuntimeError."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")

    with pytest.raises(RuntimeError):
        with normalized_wav(m4a, _run=_fake_run(returncode=1, stderr=b"codec error")):
            pass


def test_ffmpeg_error_message_mentions_original_filename(tmp_path: Path) -> None:
    """The RuntimeError must mention the original filename for user-facing clarity."""
    m4a = tmp_path / "important_recording.m4a"
    m4a.write_bytes(b"fake m4a")

    with pytest.raises(RuntimeError, match="important_recording"):
        with normalized_wav(m4a, _run=_fake_run(returncode=1)):
            pass


def test_ffmpeg_error_message_mentions_ffmpeg(tmp_path: Path) -> None:
    """The RuntimeError must mention ffmpeg so users know what tool failed."""
    m4a = tmp_path / "voice.m4a"
    m4a.write_bytes(b"fake m4a")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        with normalized_wav(m4a, _run=_fake_run(returncode=1, stderr=b"error detail")):
            pass
