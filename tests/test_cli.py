from pathlib import Path

from typer.testing import CliRunner

from meeting_protocol import __version__
from meeting_protocol.cli.main import app

runner = CliRunner()
_FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_transcript.json"


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_from_transcript_creates_files(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "from-transcript",
            str(_FIXTURE),
            "--out",
            str(tmp_path),
            "--title",
            "Test Run",
            "--participants",
            "Yogev,Tom",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "protocol.md").exists()
    assert (tmp_path / "transcript.md").exists()
    assert (tmp_path / "actions.md").exists()


def test_from_transcript_protocol_content(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "from-transcript",
            str(_FIXTURE),
            "--out",
            str(tmp_path),
            "--title",
            "Team Sync",
            "--participants",
            "Yogev,Tom",
        ],
    )
    protocol_md = (tmp_path / "protocol.md").read_text()
    assert "# Team Sync" in protocol_md
    assert "Yogev" in protocol_md
