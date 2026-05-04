import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from meeting_protocol.diarization.base import Diarizer, SpeakerInterval

PipelineFactory = Callable[[str, str | None], Any]


def _default_pipeline_factory(model: str, auth_token: str | None) -> Any:
    """Build a real pyannote.audio pipeline. Lazy-imports torch/pyannote so the
    module is importable without the optional dep installed."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            "pyannote.audio is required for diarization. "
            "Install with: pip install meeting-protocol[pyannote]"
        ) from e

    pipeline = Pipeline.from_pretrained(model, token=auth_token)
    if pipeline is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained({model!r}) returned None — "
            "this usually means the HuggingFace auth token is missing or has not been "
            "granted access to the model. See README for setup steps."
        )
    if hasattr(torch, "backends") and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    return pipeline


class PyannoteDiarizer(Diarizer):
    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        auth_token: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        token = auth_token if auth_token is not None else os.environ.get("HF_TOKEN")
        if not token and pipeline_factory is None:
            raise RuntimeError(
                "HuggingFace auth token not found. Set HF_TOKEN env var or pass "
                "--hf-token. Steps:\n"
                "  1. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  2. Accept license: https://huggingface.co/pyannote/segmentation-3.0\n"
                "  3. Create token:   https://huggingface.co/settings/tokens\n"
                "  4. export HF_TOKEN=<your-token>"
            )
        self._model = model
        self._auth_token = token
        self._num_speakers = num_speakers
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._pipeline_factory: PipelineFactory = (
            pipeline_factory if pipeline_factory is not None else _default_pipeline_factory
        )
        self._pipeline: Any | None = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(self._model, self._auth_token)
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
        pipeline = self._ensure_pipeline()
        kwargs: dict[str, int] = {}
        if self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers
        if self._min_speakers is not None:
            kwargs["min_speakers"] = self._min_speakers
        if self._max_speakers is not None:
            kwargs["max_speakers"] = self._max_speakers
        result = pipeline(str(audio_path), **kwargs)

        # pyannote.audio ≥ 4.0 wraps the output in a DiarizeOutput object;
        # older versions return an Annotation directly.
        if hasattr(result, "itertracks"):
            annotation = result
        elif (
            hasattr(result, "exclusive_speaker_diarization")
            and result.exclusive_speaker_diarization is not None
        ):
            annotation = result.exclusive_speaker_diarization
        elif hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        else:
            raise RuntimeError(f"Unexpected pipeline output type: {type(result)}")

        intervals: list[SpeakerInterval] = []
        for turn, _track, label in annotation.itertracks(yield_label=True):
            intervals.append(
                SpeakerInterval(start=float(turn.start), end=float(turn.end), label=str(label))
            )
        return intervals
