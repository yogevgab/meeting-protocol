import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from meeting_protocol.models import Segment, Transcript
from meeting_protocol.transcription.base import TranscriptionProvider

SubprocessRunner = Callable[[list[str]], str]


def _default_runner(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


class WhisperCppProvider(TranscriptionProvider):
    def __init__(
        self,
        model_path: Path,
        whisper_cli: str = "whisper-cli",
        output_dir: Path | None = None,
        runner: SubprocessRunner | None = None,
    ) -> None:
        self._model_path = model_path
        self._whisper_cli = whisper_cli
        self._output_dir = output_dir
        self._runner: SubprocessRunner = runner if runner is not None else _default_runner

    def _build_command(self, audio_path: Path) -> list[str]:
        cmd = [
            self._whisper_cli,
            "--model", str(self._model_path),
            "--output-json",
        ]
        if self._output_dir is not None:
            cmd += ["--output-file", str(self._output_dir / audio_path.stem)]
        cmd.append(str(audio_path))
        return cmd

    def transcribe(self, audio_path: Path) -> Transcript:
        cmd = self._build_command(audio_path)
        output = self._runner(cmd)
        if not output.strip() and self._output_dir is not None:
            sidecar = self._output_dir / f"{audio_path.stem}.json"
            output = sidecar.read_text(encoding="utf-8")
        data = json.loads(output)
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
