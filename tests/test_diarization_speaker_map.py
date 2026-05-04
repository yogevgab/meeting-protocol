import pytest

from meeting_protocol.diarization.speaker_map import parse_speaker_map


def test_well_formed_pair() -> None:
    assert parse_speaker_map("SPEAKER_00=Yogev") == {"SPEAKER_00": "Yogev"}


def test_well_formed_multiple_pairs() -> None:
    assert parse_speaker_map("SPEAKER_00=Yogev,SPEAKER_01=Tom") == {
        "SPEAKER_00": "Yogev",
        "SPEAKER_01": "Tom",
    }


def test_whitespace_around_pairs_is_tolerated() -> None:
    assert parse_speaker_map("  SPEAKER_00 = Yogev , SPEAKER_01 = Tom  ") == {
        "SPEAKER_00": "Yogev",
        "SPEAKER_01": "Tom",
    }


def test_empty_string_returns_empty_dict() -> None:
    assert parse_speaker_map("") == {}


def test_whitespace_only_returns_empty_dict() -> None:
    assert parse_speaker_map("   ") == {}


def test_trailing_comma_tolerated() -> None:
    assert parse_speaker_map("SPEAKER_00=Yogev,") == {"SPEAKER_00": "Yogev"}


def test_missing_equals_raises() -> None:
    with pytest.raises(ValueError):
        parse_speaker_map("SPEAKER_00 Yogev")


def test_empty_label_raises() -> None:
    with pytest.raises(ValueError):
        parse_speaker_map("=Yogev")


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError):
        parse_speaker_map("SPEAKER_00=")


def test_name_can_contain_spaces() -> None:
    assert parse_speaker_map("SPEAKER_00=Yogev Gabay") == {"SPEAKER_00": "Yogev Gabay"}
