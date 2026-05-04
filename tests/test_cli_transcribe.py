from pathlib import Path

import pytest
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


# ── whisper-cpp provider: binary path and output directory ────────────────────
#
# The CLI must accept --whisper-cli and forward both it and the --out directory
# to WhisperCppProvider so the provider can write the sidecar JSON without
# requiring anything on stdout.


def test_transcribe_whisper_cpp_accepts_whisper_cli_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--whisper-cli value must be forwarded to WhisperCppProvider as whisper_cli."""
    captured: dict[str, object] = {}

    class _FakeWhisperProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def transcribe(self, audio_path: Path) -> Transcript:
            return Transcript(segments=[], duration=0.0, source_file=str(audio_path))

    monkeypatch.setattr("meeting_protocol.cli.main.WhisperCppProvider", _FakeWhisperProvider)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(_audio(tmp_path)),
            "--provider", "whisper-cpp",
            "--model", str(tmp_path / "ggml-base.bin"),
            "--whisper-cli", "/opt/homebrew/bin/whisper-cli",
            "--out", str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert str(captured.get("whisper_cli")) == "/opt/homebrew/bin/whisper-cli"


def test_transcribe_whisper_cpp_accepts_language_option_and_forwards_to_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--language he must be forwarded to WhisperCppProvider as language='he'."""
    captured: dict[str, object] = {}

    class _FakeWhisperProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def transcribe(self, audio_path: Path) -> Transcript:
            return Transcript(segments=[], duration=0.0, source_file=str(audio_path))

    monkeypatch.setattr("meeting_protocol.cli.main.WhisperCppProvider", _FakeWhisperProvider)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(_audio(tmp_path)),
            "--provider", "whisper-cpp",
            "--model", str(tmp_path / "ggml-base.bin"),
            "--language", "he",
            "--out", str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("language") == "he"


def test_transcribe_whisper_cpp_default_language_is_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When --language is omitted, WhisperCppProvider must receive language='auto'."""
    captured: dict[str, object] = {}

    class _FakeWhisperProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def transcribe(self, audio_path: Path) -> Transcript:
            return Transcript(segments=[], duration=0.0, source_file=str(audio_path))

    monkeypatch.setattr("meeting_protocol.cli.main.WhisperCppProvider", _FakeWhisperProvider)

    result = runner.invoke(
        app,
        [
            "transcribe",
            str(_audio(tmp_path)),
            "--provider", "whisper-cpp",
            "--model", str(tmp_path / "ggml-base.bin"),
            "--out", str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("language") == "auto"


def test_transcribe_whisper_cpp_passes_out_dir_to_provider_as_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The --out directory must be forwarded to WhisperCppProvider as output_dir."""
    captured: dict[str, object] = {}

    class _FakeWhisperProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def transcribe(self, audio_path: Path) -> Transcript:
            return Transcript(segments=[], duration=0.0, source_file=str(audio_path))

    monkeypatch.setattr("meeting_protocol.cli.main.WhisperCppProvider", _FakeWhisperProvider)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe",
            str(_audio(tmp_path)),
            "--provider", "whisper-cpp",
            "--model", str(tmp_path / "ggml-base.bin"),
            "--out", str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("output_dir") == out_dir


# ── ensemble + correction flags ───────────────────────────────────────────────


def test_transcribe_correct_falls_back_gracefully_when_ollama_unreachable(
    tmp_path: Path,
) -> None:
    """--correct with a bogus host must finish, write transcript.json, and not crash."""
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--correct",
            "--correction-host", "http://127.0.0.1:1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out / "transcript.json").exists()
    assert (out / "transcript.raw.json").exists()


def test_transcribe_correct_writes_raw_and_corrected_json(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--correct",
            "--correction-host", "http://127.0.0.1:1",  # forced fallback
        ],
    )

    assert result.exit_code == 0
    raw = load_transcript(out / "transcript.raw.json")
    corrected = load_transcript(out / "transcript.json")
    assert [s.id for s in raw.segments] == [s.id for s in corrected.segments]
    # Under fallback, corrected text == raw text per segment.
    assert [s.text for s in raw.segments] == [s.text for s in corrected.segments]


def test_transcribe_no_correct_writes_only_transcript_json(tmp_path: Path) -> None:
    """Without --correct or --secondary-model, no raw/secondary files appear."""
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(_audio(tmp_path)), "--out", str(out), "--provider", "mock"],
    )

    assert result.exit_code == 0
    assert (out / "transcript.json").exists()
    assert not (out / "transcript.raw.json").exists()
    assert not (out / "transcript.secondary.json").exists()


# ── diarization flags ─────────────────────────────────────────────────────────


def test_diarize_populates_speaker_id_and_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--diarize with a stubbed PyannoteDiarizer assigns speaker_id per segment."""
    from meeting_protocol.diarization.base import Diarizer, SpeakerInterval
    from meeting_protocol.models import Segment

    class _StubDiarizer(Diarizer):
        def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
            return [
                SpeakerInterval(start=0.0, end=10.0, label="SPEAKER_00"),
                SpeakerInterval(start=10.0, end=20.0, label="SPEAKER_01"),
            ]

    def _fake_build(**_kwargs: object) -> Diarizer:
        return _StubDiarizer()

    monkeypatch.setattr("meeting_protocol.cli.main._build_diarizer", _fake_build)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--diarize",
        ],
    )

    assert result.exit_code == 0, result.output
    transcript = load_transcript(out / "transcript.json")
    assert all(isinstance(s, Segment) for s in transcript.segments)
    speaker_ids = {s.speaker_id for s in transcript.segments}
    assert speaker_ids.issubset({"SPEAKER_00", "SPEAKER_01"})
    assert len(transcript.speakers) >= 1


def test_diarize_with_speaker_map_renames_in_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meeting_protocol.diarization.base import Diarizer, SpeakerInterval

    class _StubDiarizer(Diarizer):
        def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
            return [SpeakerInterval(start=0.0, end=100.0, label="SPEAKER_00")]

    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _StubDiarizer(),
    )

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--diarize",
            "--speaker-map", "SPEAKER_00=Yogev",
        ],
    )

    assert result.exit_code == 0, result.output
    transcript = load_transcript(out / "transcript.json")
    speaker_names = {sp.id: sp.name for sp in transcript.speakers}
    assert speaker_names.get("SPEAKER_00") == "Yogev"
    md = (out / "transcript.md").read_text(encoding="utf-8")
    assert "Yogev" in md


def test_diarize_writes_raw_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When --diarize is on, transcript.raw.json is written (post-stage marker)."""
    from meeting_protocol.diarization.base import Diarizer, SpeakerInterval

    class _StubDiarizer(Diarizer):
        def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
            return [SpeakerInterval(0.0, 100.0, "SPEAKER_00")]

    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _StubDiarizer(),
    )

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--diarize",
        ],
    )

    assert result.exit_code == 0
    assert (out / "transcript.raw.json").exists()


def test_diarize_without_token_fails_fast_with_setup_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--diarize without HF_TOKEN env var or --hf-token errors with HF setup URLs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--diarize",
        ],
    )

    assert result.exit_code != 0
    assert "HF_TOKEN" in result.output or "huggingface" in result.output.lower()


def test_diarize_plus_correct_preserves_speaker_id_through_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: --diarize then --correct must yield a transcript whose
    segments have BOTH speaker_id (from diarization) AND finalized text
    (from correction). The fallback path is fine — we just need the IDs to survive."""
    from meeting_protocol.diarization.base import Diarizer, SpeakerInterval

    class _StubDiarizer(Diarizer):
        def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
            return [SpeakerInterval(0.0, 100.0, "SPEAKER_00")]

    monkeypatch.setattr(
        "meeting_protocol.cli.main._build_diarizer",
        lambda **_kw: _StubDiarizer(),
    )

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--diarize",
            "--speaker-map", "SPEAKER_00=Yogev",
            "--correct",
            "--correction-host", "http://127.0.0.1:1",  # forces correction fallback
        ],
    )

    assert result.exit_code == 0, result.output
    final = load_transcript(out / "transcript.json")
    raw = load_transcript(out / "transcript.raw.json")
    # speaker_id survived correction (fallback or otherwise)
    assert all(s.speaker_id == "SPEAKER_00" for s in final.segments)
    assert {sp.id: sp.name for sp in final.speakers} == {"SPEAKER_00": "Yogev"}
    # raw and final agree on speaker assignment
    assert [s.speaker_id for s in raw.segments] == [s.speaker_id for s in final.segments]


def test_diarize_with_invalid_speaker_map_exits_nonzero(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--diarize",
            "--speaker-map", "this is malformed",
        ],
    )

    assert result.exit_code != 0


def test_transcribe_secondary_model_writes_secondary_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--secondary-model triggers ensemble; transcript.secondary.json gets written."""
    out = tmp_path / "out"
    received_models: list[str] = []

    class _FakeWhisperProvider:
        def __init__(self, **kwargs: object) -> None:
            received_models.append(str(kwargs["model_path"]))

        def transcribe(self, audio_path: Path) -> Transcript:
            from meeting_protocol.models import Segment
            return Transcript(
                segments=[Segment(id=1, start=0.0, end=1.0, text="x")],
                duration=1.0,
                source_file=str(audio_path),
            )

    monkeypatch.setattr("meeting_protocol.cli.main.WhisperCppProvider", _FakeWhisperProvider)

    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--provider", "whisper-cpp",
            "--model", str(tmp_path / "primary.bin"),
            "--secondary-model", str(tmp_path / "secondary.bin"),
            "--out", str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out / "transcript.secondary.json").exists()
    assert (out / "transcript.raw.json").exists()
    assert str(tmp_path / "primary.bin") in received_models
    assert str(tmp_path / "secondary.bin") in received_models


# ── --correction-think option ─────────────────────────────────────────────────


def test_correction_think_false_passes_think_false_to_corrector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--correction-think false must wire think=False into the OllamaCorrector."""
    from meeting_protocol.models import Transcript

    captured: dict[str, object] = {}

    class _FakeCorrector:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def correct(self, transcript: Transcript, secondary_texts: object = None) -> Transcript:
            return transcript

    monkeypatch.setattr("meeting_protocol.cli.main.OllamaCorrector", _FakeCorrector)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--correct",
            "--correction-think", "false",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("think") is False


def test_correction_think_true_passes_think_true_to_corrector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--correction-think true must wire think=True into the OllamaCorrector."""
    from meeting_protocol.models import Transcript

    captured: dict[str, object] = {}

    class _FakeCorrector:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def correct(self, transcript: Transcript, secondary_texts: object = None) -> Transcript:
            return transcript

    monkeypatch.setattr("meeting_protocol.cli.main.OllamaCorrector", _FakeCorrector)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--correct",
            "--correction-think", "true",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("think") is True


def test_correction_think_invalid_value_exits_nonzero_with_helpful_message(
    tmp_path: Path,
) -> None:
    """--correction-think with an unknown value must exit non-zero and name --correction-think."""
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(tmp_path / "out"),
            "--provider", "mock",
            "--correction-think", "maybe",
        ],
    )

    assert result.exit_code != 0
    assert "--correction-think" in result.output


def test_correction_think_auto_omits_think_from_corrector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--correction-think auto (default) must pass think=None to OllamaCorrector."""
    from meeting_protocol.models import Transcript

    captured: dict[str, object] = {}

    class _FakeCorrector:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def correct(self, transcript: Transcript, secondary_texts: object = None) -> Transcript:
            return transcript

    monkeypatch.setattr("meeting_protocol.cli.main.OllamaCorrector", _FakeCorrector)

    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "transcribe", str(_audio(tmp_path)),
            "--out", str(out),
            "--provider", "mock",
            "--correct",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("think") is None
