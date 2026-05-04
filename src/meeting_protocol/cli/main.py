from pathlib import Path
from typing import Annotated

import typer

from meeting_protocol import __version__
from meeting_protocol.io import load_transcript, save_transcript
from meeting_protocol.outputs.markdown import render_actions, render_protocol, render_transcript
from meeting_protocol.protocol.generator import generate_protocol
from meeting_protocol.transcription.base import TranscriptionProvider
from meeting_protocol.transcription.mock import MockTranscriptionProvider
from meeting_protocol.transcription.whisper_cpp import WhisperCppProvider

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


@app.command()
def transcribe(
    audio_path: Annotated[Path, typer.Argument(help="Path to audio file.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output directory.")
    ] = Path("."),
    provider: Annotated[
        str, typer.Option("--provider", help="Transcription backend: mock or whisper-cpp.")
    ] = "mock",
    model: Annotated[
        str, typer.Option("--model", help="Path to whisper.cpp model (whisper-cpp only).")
    ] = "",
    whisper_cli: Annotated[
        str, typer.Option("--whisper-cli", help="Path to whisper-cli binary (whisper-cpp only).")
    ] = "whisper-cli",
    language: Annotated[
        str, typer.Option("--language", help="Language code for transcription (whisper-cpp only).")
    ] = "auto",
    title: Annotated[
        str, typer.Option("--title", "-t", help="Protocol title.")
    ] = "Meeting Protocol",
    participants: Annotated[
        str, typer.Option("--participants", "-p", help="Comma-separated participant names.")
    ] = "",
) -> None:
    """Transcribe audio and write JSON + Markdown artifacts."""
    if not audio_path.exists():
        typer.echo(f"Error: audio file not found: {audio_path}", err=True)
        raise typer.Exit(code=1)

    backend: TranscriptionProvider
    if provider == "mock":
        backend = MockTranscriptionProvider()
    elif provider == "whisper-cpp":
        if not model:
            typer.echo("Error: --model is required for whisper-cpp provider.", err=True)
            raise typer.Exit(code=1)
        backend = WhisperCppProvider(
            model_path=Path(model),
            whisper_cli=whisper_cli,
            language=language,
            output_dir=out,
        )
    else:
        typer.echo(f"Error: unknown provider {provider!r}. Choose mock or whisper-cpp.", err=True)
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    transcript = backend.transcribe(audio_path)
    save_transcript(transcript, out / "transcript.json")

    participant_list = [p.strip() for p in participants.split(",") if p.strip()]
    protocol = generate_protocol(transcript, title=title, participants=participant_list)
    (out / "protocol.md").write_text(render_protocol(protocol), encoding="utf-8")
    (out / "transcript.md").write_text(render_transcript(transcript), encoding="utf-8")
    (out / "actions.md").write_text(render_actions(protocol), encoding="utf-8")
    typer.echo(f"Wrote transcript.json, protocol.md, transcript.md, actions.md → {out}")
