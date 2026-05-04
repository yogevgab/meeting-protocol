import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from meeting_protocol.diarization.pyannote_diarizer import (
    PyannoteDiarizer,
    _default_pipeline_factory,
)


@dataclass(frozen=True)
class _StubTurn:
    start: float
    end: float


class _StubAnnotation:
    """Stands in for a pyannote Annotation. Its itertracks(yield_label=True)
    yields (turn, track_id, label) triples — same shape as the real thing."""

    def __init__(self, triples: Iterable[tuple[_StubTurn, str, str]]) -> None:
        self._triples = list(triples)

    def itertracks(
        self, yield_label: bool = False
    ) -> Iterable[tuple[_StubTurn, str, str]]:
        assert yield_label is True
        return iter(self._triples)


class _StubPipeline:
    """Callable pipeline that records its inputs and returns a canned annotation."""

    def __init__(self, annotation: _StubAnnotation) -> None:
        self._annotation = annotation
        self.calls: list[tuple[str, dict[str, int]]] = []

    def __call__(self, audio_path: str, **kwargs: int) -> _StubAnnotation:
        self.calls.append((audio_path, dict(kwargs)))
        return self._annotation


def _factory(pipeline: _StubPipeline) -> Any:
    received: list[tuple[str, str | None]] = []

    def factory(model: str, auth_token: str | None) -> _StubPipeline:
        received.append((model, auth_token))
        return pipeline

    factory.received = received  # type: ignore[attr-defined]
    return factory


# ── happy path ────────────────────────────────────────────────────────────────


def test_diarize_returns_intervals_from_annotation(tmp_path: Path) -> None:
    annotation = _StubAnnotation(
        [
            (_StubTurn(0.0, 5.0), "track1", "SPEAKER_00"),
            (_StubTurn(5.0, 10.0), "track2", "SPEAKER_01"),
        ]
    )
    pipeline = _StubPipeline(annotation)
    d = PyannoteDiarizer(auth_token="t", pipeline_factory=_factory(pipeline))

    intervals = d.diarize(tmp_path / "audio.wav")

    assert [(i.start, i.end, i.label) for i in intervals] == [
        (0.0, 5.0, "SPEAKER_00"),
        (5.0, 10.0, "SPEAKER_01"),
    ]


def test_factory_receives_configured_model_and_token(tmp_path: Path) -> None:
    pipeline = _StubPipeline(_StubAnnotation([]))
    factory = _factory(pipeline)
    d = PyannoteDiarizer(
        model="my/model:tag", auth_token="hf_xxx", pipeline_factory=factory
    )

    d.diarize(tmp_path / "audio.wav")

    assert factory.received == [("my/model:tag", "hf_xxx")]


def test_pipeline_built_lazily_and_only_once(tmp_path: Path) -> None:
    pipeline = _StubPipeline(_StubAnnotation([]))
    factory = _factory(pipeline)
    d = PyannoteDiarizer(auth_token="t", pipeline_factory=factory)

    d.diarize(tmp_path / "a.wav")
    d.diarize(tmp_path / "b.wav")

    assert len(factory.received) == 1


# ── num_speakers / min_speakers / max_speakers passthrough ────────────────────


def test_num_speakers_is_passed_to_pipeline(tmp_path: Path) -> None:
    pipeline = _StubPipeline(_StubAnnotation([]))
    d = PyannoteDiarizer(
        auth_token="t", num_speakers=2, pipeline_factory=_factory(pipeline)
    )

    d.diarize(tmp_path / "a.wav")

    assert pipeline.calls[0][1] == {"num_speakers": 2}


def test_min_max_speakers_are_passed_to_pipeline(tmp_path: Path) -> None:
    pipeline = _StubPipeline(_StubAnnotation([]))
    d = PyannoteDiarizer(
        auth_token="t",
        min_speakers=2,
        max_speakers=4,
        pipeline_factory=_factory(pipeline),
    )

    d.diarize(tmp_path / "a.wav")

    assert pipeline.calls[0][1] == {"min_speakers": 2, "max_speakers": 4}


def test_no_speaker_hint_passes_no_kwargs(tmp_path: Path) -> None:
    pipeline = _StubPipeline(_StubAnnotation([]))
    d = PyannoteDiarizer(auth_token="t", pipeline_factory=_factory(pipeline))

    d.diarize(tmp_path / "a.wav")

    assert pipeline.calls[0][1] == {}


# ── audio path is passed as string ───────────────────────────────────────────


def test_pipeline_receives_audio_path_as_string(tmp_path: Path) -> None:
    pipeline = _StubPipeline(_StubAnnotation([]))
    d = PyannoteDiarizer(auth_token="t", pipeline_factory=_factory(pipeline))

    audio = tmp_path / "audio.wav"
    d.diarize(audio)

    assert pipeline.calls[0][0] == str(audio)


# ── auth-token resolution ─────────────────────────────────────────────────────


def test_missing_token_without_factory_raises_with_setup_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError) as ei:
        PyannoteDiarizer()

    msg = str(ei.value)
    assert "HF_TOKEN" in msg
    assert "huggingface.co" in msg.lower()


def test_missing_token_is_ok_when_factory_is_injected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test path: an injected factory bypasses the auth precheck."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    pipeline = _StubPipeline(_StubAnnotation([]))
    d = PyannoteDiarizer(pipeline_factory=_factory(pipeline))

    intervals = d.diarize(tmp_path / "a.wav")

    assert intervals == []


def test_env_var_used_when_no_explicit_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    pipeline = _StubPipeline(_StubAnnotation([]))
    factory = _factory(pipeline)

    d = PyannoteDiarizer(pipeline_factory=factory)
    d.diarize(tmp_path / "a.wav")

    assert factory.received[0][1] == "from-env"


def test_explicit_token_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    pipeline = _StubPipeline(_StubAnnotation([]))
    factory = _factory(pipeline)

    d = PyannoteDiarizer(auth_token="explicit", pipeline_factory=factory)
    d.diarize(tmp_path / "a.wav")

    assert factory.received[0][1] == "explicit"


# ── _default_pipeline_factory unit tests ─────────────────────────────────────


def _mock_pyannote_modules(pipeline_return_value: Any) -> tuple[MagicMock, dict[str, Any]]:
    """Build mock torch/pyannote modules for patching sys.modules.

    Returns (Pipeline class mock, dict to pass to patch.dict(sys.modules, ...)).
    """
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False

    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = pipeline_return_value

    mock_pyannote = MagicMock()
    mock_pyannote_audio = MagicMock()
    mock_pyannote_audio.Pipeline = mock_pipeline_cls

    return mock_pipeline_cls, {
        "torch": mock_torch,
        "pyannote": mock_pyannote,
        "pyannote.audio": mock_pyannote_audio,
    }


def test_default_pipeline_factory_raises_runtime_error_when_pretrained_returns_none() -> None:
    pipeline_cls, modules = _mock_pyannote_modules(pipeline_return_value=None)

    with patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError, match="returned None"):
            _default_pipeline_factory("pyannote/speaker-diarization-3.1", "hf_xxx")


def test_default_pipeline_factory_calls_from_pretrained_with_token_not_use_auth_token() -> None:
    pipeline_instance = MagicMock()
    pipeline_cls, modules = _mock_pyannote_modules(pipeline_return_value=pipeline_instance)

    with patch.dict(sys.modules, modules):
        _default_pipeline_factory("my/model", "my-token")

    pipeline_cls.from_pretrained.assert_called_once_with("my/model", token="my-token")
