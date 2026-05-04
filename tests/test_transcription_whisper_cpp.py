import contextlib
import json
from collections.abc import Callable, Generator
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


def test_build_command_includes_language_flag_for_hebrew(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        language="he",
        runner=_fixed_runner(""),
    )
    cmd = provider._build_command(tmp_path / "audio.wav")

    assert "--language" in cmd
    assert cmd[cmd.index("--language") + 1] == "he"


def test_build_command_includes_language_flag_for_auto(tmp_path: Path) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        language="auto",
        runner=_fixed_runner(""),
    )
    cmd = provider._build_command(tmp_path / "audio.wav")

    assert "--language" in cmd
    assert cmd[cmd.index("--language") + 1] == "auto"


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


# ── whisper.cpp 1.8.x sidecar-file behavior ──────────────────────────────────
#
# whisper-cli --output-json --output-file <base> writes JSON to <base>.json;
# stdout is typically empty. The provider must emit the right flags and then
# fall back to reading the sidecar file when stdout is empty.


def test_build_command_includes_output_file_flag_when_output_dir_configured(
    tmp_path: Path,
) -> None:
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        output_dir=tmp_path / "out",
        runner=_fixed_runner(""),
    )
    cmd = provider._build_command(tmp_path / "meeting.wav")

    assert "--output-file" in cmd


def test_build_command_output_file_value_is_audio_stem_inside_output_dir(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    audio = tmp_path / "meeting.wav"
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        output_dir=output_dir,
        runner=_fixed_runner(""),
    )
    cmd = provider._build_command(audio)

    idx = cmd.index("--output-file")
    assert cmd[idx + 1] == str(output_dir / audio.stem)


def test_transcribe_reads_json_from_sidecar_file_when_stdout_is_empty(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    audio = tmp_path / "meeting.wav"
    audio.touch()
    (output_dir / "meeting.json").write_text(_SAMPLE_OUTPUT, encoding="utf-8")

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        output_dir=output_dir,
        runner=_fixed_runner(""),
    )
    transcript = provider.transcribe(audio)

    assert len(transcript.segments) == 2


def test_transcribe_reads_sidecar_when_stdout_contains_non_json_text(
    tmp_path: Path,
) -> None:
    """Ivrit whisper-cli can emit plain transcript text while writing JSON sidecar."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    audio = tmp_path / "meeting.wav"
    audio.touch()
    (output_dir / "meeting.json").write_text(_SAMPLE_OUTPUT, encoding="utf-8")

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        output_dir=output_dir,
        runner=_fixed_runner("Hello, this is a test.\nשלום, זה בדיקה.\n"),
    )
    transcript = provider.transcribe(audio)

    assert len(transcript.segments) == 2


def test_empty_transcription_list_yields_zero_segments(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    audio = tmp_path / "silence.wav"
    audio.touch()
    (output_dir / "silence.json").write_text(
        json.dumps({"transcription": []}), encoding="utf-8"
    )

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        output_dir=output_dir,
        runner=_fixed_runner(""),
    )
    transcript = provider.transcribe(audio)

    assert transcript.segments == []


def test_transcribe_accepts_raw_control_chars_in_segment_text(tmp_path: Path) -> None:
    """whisper-cli sometimes embeds raw newlines inside "text" values, producing
    JSON that strict parsers reject. The provider must accept such output."""
    raw = (
        '{"transcription": ['
        '{"timestamps": {"from": "00:00:00,000", "to": "00:00:01,000"},'
        ' "offsets": {"from": 0, "to": 1000},'
        ' "text": " line one\nline two"}'
        "]}"
    )
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(raw),
    )
    transcript = provider.transcribe(tmp_path / "audio.wav")

    assert len(transcript.segments) == 1
    assert "line one" in transcript.segments[0].text
    assert "line two" in transcript.segments[0].text


def test_empty_transcription_list_yields_duration_zero(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    audio = tmp_path / "silence.wav"
    audio.touch()
    (output_dir / "silence.json").write_text(
        json.dumps({"transcription": []}), encoding="utf-8"
    )

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        output_dir=output_dir,
        runner=_fixed_runner(""),
    )
    transcript = provider.transcribe(audio)

    assert transcript.duration == 0.0


# ── M4A / compressed-format normalization ─────────────────────────────────────
#
# WhisperCppProvider must normalize non-WAV inputs to WAV before invoking
# whisper-cli.  A caller-injectable normalizer makes this testable without
# spawning ffmpeg.


def _capturing_normalizer(
    tmp_path: Path,
) -> tuple[list[tuple[Path, Path]], Callable[[Path], "contextlib.AbstractContextManager[Path]"]]:
    """Return (calls_list, normalizer).  Each call records (original, yielded)."""
    calls: list[tuple[Path, Path]] = []

    @contextlib.contextmanager
    def normalizer(path: Path) -> Generator[Path, None, None]:
        wav = tmp_path / "norm" / (path.stem + ".wav")
        wav.parent.mkdir(exist_ok=True)
        wav.write_bytes(b"fake wav")
        calls.append((path, wav))
        yield wav

    return calls, normalizer


def test_transcribe_m4a_runner_receives_wav_path_not_m4a(tmp_path: Path) -> None:
    """When an M4A is given, whisper-cli must receive a .wav path, never .m4a."""
    m4a = tmp_path / "recording.m4a"
    runner_args: list[list[str]] = []

    def capturing_runner(cmd: list[str]) -> str:
        runner_args.append(cmd)
        return _SAMPLE_OUTPUT

    _, norm = _capturing_normalizer(tmp_path)
    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=capturing_runner,
        normalizer=norm,
    )
    provider.transcribe(m4a)

    assert runner_args, "runner was never called"
    cmd = runner_args[0]
    audio_args = [a for a in cmd if a.endswith((".wav", ".m4a", ".mp3", ".aac"))]
    assert audio_args, "runner received no audio path"
    assert all(a.endswith(".wav") for a in audio_args), (
        f"runner received non-WAV path: {audio_args}"
    )


def test_transcribe_m4a_source_file_is_original_m4a_path(tmp_path: Path) -> None:
    """Transcript.source_file must be the original M4A path, not the temp WAV."""
    m4a = tmp_path / "recording.m4a"
    _, norm = _capturing_normalizer(tmp_path)

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
        normalizer=norm,
    )
    transcript = provider.transcribe(m4a)

    assert transcript.source_file == str(m4a)


def test_transcribe_wav_normalizer_receives_wav_and_yields_it_unchanged(
    tmp_path: Path,
) -> None:
    """For WAV input the normalizer is called; it must yield the same WAV path."""
    wav = tmp_path / "audio.wav"
    normalizer_inputs: list[Path] = []
    normalizer_outputs: list[Path] = []
    runner_args: list[list[str]] = []

    @contextlib.contextmanager
    def tracking_normalizer(path: Path) -> Generator[Path, None, None]:
        normalizer_inputs.append(path)
        yield path  # passthrough – no conversion
        normalizer_outputs.append(path)

    def capturing_runner(cmd: list[str]) -> str:
        runner_args.append(cmd)
        return _SAMPLE_OUTPUT

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=capturing_runner,
        normalizer=tracking_normalizer,
    )
    provider.transcribe(wav)

    assert normalizer_inputs == [wav], "normalizer not called with WAV path"
    assert str(wav) in runner_args[0], "runner did not receive the WAV path"


def test_normalizer_receives_original_audio_path(tmp_path: Path) -> None:
    """The injected normalizer must receive the exact audio_path passed to transcribe()."""
    m4a = tmp_path / "voice.m4a"
    normalizer_inputs: list[Path] = []

    @contextlib.contextmanager
    def tracking_normalizer(path: Path) -> Generator[Path, None, None]:
        normalizer_inputs.append(path)
        wav = tmp_path / (path.stem + ".wav")
        wav.write_bytes(b"fake")
        yield wav

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
        normalizer=tracking_normalizer,
    )
    provider.transcribe(m4a)

    assert normalizer_inputs == [m4a]


def test_normalizer_failure_propagates_from_transcribe(tmp_path: Path) -> None:
    """If the normalizer raises, the error must propagate out of transcribe()."""
    m4a = tmp_path / "voice.m4a"

    @contextlib.contextmanager
    def failing_normalizer(path: Path) -> Generator[Path, None, None]:
        raise RuntimeError("ffmpeg normalization failed: codec not found")
        yield Path("/unreachable")

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
        normalizer=failing_normalizer,
    )
    with pytest.raises(RuntimeError, match="ffmpeg"):
        provider.transcribe(m4a)


def test_transcribe_called_twice_normalizes_each_time(tmp_path: Path) -> None:
    """Each transcribe() call must invoke the normalizer once, independently."""
    calls, norm = _capturing_normalizer(tmp_path)

    provider = WhisperCppProvider(
        model_path=tmp_path / "ggml-base.bin",
        runner=_fixed_runner(_SAMPLE_OUTPUT),
        normalizer=norm,
    )
    m4a1 = tmp_path / "clip1.m4a"
    m4a2 = tmp_path / "clip2.m4a"
    provider.transcribe(m4a1)
    provider.transcribe(m4a2)

    originals = [orig for orig, _ in calls]
    assert m4a1 in originals
    assert m4a2 in originals
    assert len(calls) == 2
