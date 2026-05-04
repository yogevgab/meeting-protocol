from pathlib import Path
from typing import Annotated

import typer

from meeting_protocol import __version__
from meeting_protocol.io import load_transcript
from meeting_protocol.outputs.markdown import render_actions, render_protocol, render_transcript
from meeting_protocol.protocol.generator import generate_protocol

app = typer.Typer(name="meeting-protocol", help="Meeting transcription and protocol generator.")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command(name="from-transcript")
def from_transcript(
    transcript_path: Annotated[Path, typer.Argument(help="Path to JSON transcript file.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output directory.")
    ] = Path("."),
    title: Annotated[
        str, typer.Option("--title", "-t", help="Protocol title.")
    ] = "Meeting Protocol",
    participants: Annotated[
        str, typer.Option("--participants", "-p", help="Comma-separated participant names.")
    ] = "",
) -> None:
    """Load a JSON transcript and write Markdown protocol, transcript, and actions files."""
    participant_list = [p.strip() for p in participants.split(",") if p.strip()]
    out.mkdir(parents=True, exist_ok=True)
    transcript = load_transcript(transcript_path)
    protocol = generate_protocol(transcript, title=title, participants=participant_list)
    (out / "protocol.md").write_text(render_protocol(protocol), encoding="utf-8")
    (out / "transcript.md").write_text(render_transcript(transcript), encoding="utf-8")
    (out / "actions.md").write_text(render_actions(protocol), encoding="utf-8")
    typer.echo(f"Wrote protocol.md, transcript.md, actions.md → {out}")
