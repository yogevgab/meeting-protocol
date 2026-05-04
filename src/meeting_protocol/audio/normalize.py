import contextlib
import subprocess
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

_WAV_SUFFIXES = frozenset({".wav"})


@contextlib.contextmanager
def normalized_wav(
    path: Path,
    *,
    _run: Callable[..., Any] = subprocess.run,
) -> Generator[Path, None, None]:
    """Context manager yielding a WAV path suitable for whisper-cli.

    If *path* is already a WAV file it is yielded unchanged and ffmpeg is never
    called.  Otherwise a temporary mono 16 kHz PCM WAV is produced via ffmpeg
    and cleaned up when the context exits.

    Raises RuntimeError if ffmpeg exits non-zero.
    """
    if path.suffix.lower() in _WAV_SUFFIXES:
        yield path
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / (path.stem + ".wav")
        result: subprocess.CompletedProcess[bytes] = _run(
            [
                "ffmpeg",
                "-y",
                "-i", str(path),
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg normalization failed for {path.name} "
                f"(exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace')}"
            )
        yield wav_path
