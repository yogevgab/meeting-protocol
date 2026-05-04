import json
import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from meeting_protocol.audio.normalize import normalized_wav
from meeting_protocol.models import Segment, Transcript
from meeting_protocol.transcription.base import TranscriptionProvider

SubprocessRunner = Callable[[list[str]], str]
Normalizer = Callable[[Path], "AbstractContextManager[Path]"]


def _default_runner(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


class WhisperCppProvider(TranscriptionProvider):
    def __init__(
        self,
        model_path: Path,
        whisper_cli: str = "whisper-cli",
        language: str = "auto",
        output_dir: Path | None = None,
        runner: SubprocessRunner | None = None,
        normalizer: Normalizer | None = None,
    ) -> None:
        self._model_path = model_path
        self._whisper_cli = whisper_cli
        self._language = language
        self._output_dir = output_dir
        self._runner: SubprocessRunner = runner if runner is not None else _default_runner
        self._normalizer: Normalizer = normalizer if normalizer is not None else normalized_wav

    def _build_command(self, audio_path: Path) -> list[str]:
        cmd = [
            self._whisper_cli,
            "--model", str(self._model_path),
            "--output-json",
            "--language", self._language,
        ]
        if self._output_dir is not None:
            cmd += ["--output-file", str(self._output_dir / audio_path.stem)]
        cmd.append(str(audio_path))
        return cmd

    def transcribe(self, audio_path: Path) -> Transcript:
        with self._normalizer(audio_path) as wav_path:
            cmd = self._build_command(wav_path)
            output = self._runner(cmd)
            if self._output_dir is not None:
                sidecar = self._output_dir / f"{wav_path.stem}.json"
                use_sidecar = not output.strip()
                if not use_sidecar:
                    try:
                        json.loads(output, strict=False)
                    except (json.JSONDecodeError, ValueError):
                        use_sidecar = True
                if use_sidecar and sidecar.exists():
                    output = sidecar.read_text(encoding="utf-8")
            # strict=False: whisper-cli occasionally embeds raw newlines in transcribed text.
            data = json.loads(output, strict=False)
        segments: list[Segment] = []
        for i, raw in enumerate(data["transcription"], start=1):
            start = raw["offsets"]["from"] / 1000.0
            end = raw["offsets"]["to"] / 1000.0
            text = raw["text"].strip()
            segments.append(Segment(id=i, start=start, end=end, text=text))
        duration = max(seg.end for seg in segments) if segments else 0.0
        return Transcript(
            segments=segments,
            duration=duration,
            source_file=str(audio_path),
        )
