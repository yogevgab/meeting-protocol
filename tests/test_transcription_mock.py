from pathlib import Path

import pytest

from meeting_protocol.models import Segment, Speaker, Transcript
from meeting_protocol.transcription.mock import MockTranscriptionProvider


@pytest.fixture()
def audio_file(tmp_path: Path) -> Path:
    f = tmp_path / "meeting.mp3"
    f.write_bytes(b"fake audio data")
    return f


def test_mock_provider_returns_transcript(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    assert isinstance(result, Transcript)


def test_mock_provider_source_file_matches_audio_path(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    assert result.source_file == str(audio_file)


def test_mock_provider_has_positive_duration(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    assert result.duration > 0


def test_mock_provider_has_at_least_one_segment(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    assert len(result.segments) > 0


def test_mock_provider_segments_have_valid_structure(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    for seg in result.segments:
        assert isinstance(seg, Segment)
        assert seg.start >= 0
        assert seg.end > seg.start
        assert seg.text.strip()


def test_mock_provider_has_speaker_metadata(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    assert len(result.speakers) >= 1
    for speaker in result.speakers:
        assert isinstance(speaker, Speaker)
        assert speaker.id
        assert speaker.name


def test_mock_provider_segments_reference_known_speakers(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    speaker_ids = {s.id for s in result.speakers}
    for seg in result.segments:
        assert seg.speaker_id in speaker_ids, (
            f"segment speaker_id={seg.speaker_id!r} not in transcript.speakers"
        )


def test_mock_provider_duration_matches_last_segment_end(audio_file: Path) -> None:
    provider = MockTranscriptionProvider()
    result = provider.transcribe(audio_file)

    last_end = max(seg.end for seg in result.segments)
    assert result.duration == pytest.approx(last_end)
