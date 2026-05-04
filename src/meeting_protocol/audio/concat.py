import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def concat_audio_files(
    paths: list[Path],
    output: Path,
    *,
    _run: Callable[..., Any] = subprocess.run,
) -> None:
    """Concatenate audio files into one diarization-ready WAV.

    Requires ffmpeg on PATH. The output is always normalized to mono 16 kHz
    PCM WAV so downstream diarization receives a predictable continuous audio
    timeline even when inputs are compressed formats such as M4A/AAC.
    Raises RuntimeError if ffmpeg exits non-zero. The temporary filelist is
    always cleaned up, even on failure.
    """
    list_file = output.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file {_quote_concat_path(p.resolve())}" for p in paths),
        encoding="utf-8",
    )
    try:
        result: subprocess.CompletedProcess[bytes] = _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            capture_output=True,
        )
    finally:
        list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )


def _quote_concat_path(path: Path) -> str:
    """Quote a path for ffmpeg concat-demuxer file lists."""
    return "'" + str(path).replace("'", "'\\''") + "'"
