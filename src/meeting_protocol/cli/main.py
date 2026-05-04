from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Annotated

import typer

from meeting_protocol import __version__
from meeting_protocol.correction.ollama import OllamaCorrector
from meeting_protocol.diarization.assignment import assign_speakers
from meeting_protocol.diarization.base import Diarizer
from meeting_protocol.diarization.pyannote_diarizer import PyannoteDiarizer
from meeting_protocol.diarization.speaker_map import parse_speaker_map
from meeting_protocol.io import load_transcript, save_transcript
from meeting_protocol.outputs.markdown import render_actions, render_protocol, render_transcript
from meeting_protocol.protocol.generator import generate_protocol
from meeting_protocol.transcription.base import TranscriptionProvider
from meeting_protocol.transcription.ensemble import transcribe_ensemble
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
    secondary_model: Annotated[
        str,
        typer.Option(
            "--secondary-model",
            help="Path to a second whisper.cpp model. Enables ensemble transcription.",
        ),
    ] = "",
    alignment_iou: Annotated[
        float,
        typer.Option(
            "--alignment-iou",
            help="Minimum IoU for accepting a secondary alignment per primary segment.",
        ),
    ] = 0.3,
    correct: Annotated[
        bool,
        typer.Option(
            "--correct/--no-correct",
            help="Enable LLM post-correction of the transcript.",
        ),
    ] = False,
    corrector: Annotated[
        str, typer.Option("--corrector", help="Correction backend: ollama.")
    ] = "ollama",
    correction_model: Annotated[
        str, typer.Option("--correction-model", help="LLM model tag (e.g. gemma3:12b).")
    ] = "gemma3:12b",
    correction_host: Annotated[
        str, typer.Option("--correction-host", help="Ollama HTTP host.")
    ] = "http://localhost:11434",
    correction_window: Annotated[
        int, typer.Option("--correction-window", help="Segments per LLM call.")
    ] = 20,
    correction_overlap: Annotated[
        int,
        typer.Option(
            "--correction-overlap",
            help="Number of previously-corrected segments to include as context.",
        ),
    ] = 4,
    diarize: Annotated[
        bool,
        typer.Option(
            "--diarize/--no-diarize",
            help="Enable speaker diarization (assign speaker_id per segment).",
        ),
    ] = False,
    diarizer: Annotated[
        str, typer.Option("--diarizer", help="Diarization backend: pyannote.")
    ] = "pyannote",
    num_speakers: Annotated[
        int,
        typer.Option(
            "--num-speakers",
            help="Exact speaker count (hard constraint). 0 = auto-detect.",
        ),
    ] = 0,
    min_speakers: Annotated[
        int,
        typer.Option("--min-speakers", help="Lower bound for speaker auto-detect (0 = unset)."),
    ] = 0,
    max_speakers: Annotated[
        int,
        typer.Option("--max-speakers", help="Upper bound for speaker auto-detect (0 = unset)."),
    ] = 0,
    speaker_map: Annotated[
        str,
        typer.Option(
            "--speaker-map",
            help="Rename anonymous labels, e.g. 'SPEAKER_00=Yogev,SPEAKER_01=Tom'.",
        ),
    ] = "",
    hf_token: Annotated[
        str,
        typer.Option(
            "--hf-token",
            help="HuggingFace auth token. Defaults to $HF_TOKEN. Required for pyannote.",
        ),
    ] = "",
) -> None:
    """Transcribe audio and write JSON + Markdown artifacts."""
    if not audio_path.exists():
        typer.echo(f"Error: audio file not found: {audio_path}", err=True)
        raise typer.Exit(code=1)

    primary_backend = _build_primary_backend(provider, model, whisper_cli, language, out)

    secondary_backend: TranscriptionProvider | None = None
    if secondary_model:
        secondary_dir = out / "secondary"
        secondary_dir.mkdir(parents=True, exist_ok=True)
        secondary_backend = WhisperCppProvider(
            model_path=Path(secondary_model),
            whisper_cli=whisper_cli,
            language=language,
            output_dir=secondary_dir,
        )

    out.mkdir(parents=True, exist_ok=True)
    ensemble = transcribe_ensemble(
        primary_backend, secondary_backend, audio_path, iou_threshold=alignment_iou
    )

    transcript = ensemble.primary
    participant_list = [p.strip() for p in participants.split(",") if p.strip()]

    if diarize:
        try:
            speaker_map_dict = parse_speaker_map(speaker_map)
        except ValueError as e:
            typer.echo(f"Error: invalid --speaker-map: {e}", err=True)
            raise typer.Exit(code=1) from None
        diarizer_obj = _build_diarizer(
            name=diarizer,
            hf_token=hf_token,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            participants=participant_list,
        )
        try:
            intervals = diarizer_obj.diarize(audio_path)
        except (RuntimeError, ImportError) as e:
            typer.echo(f"Error: diarization failed: {e}", err=True)
            raise typer.Exit(code=1) from None
        new_segs, speakers = assign_speakers(
            transcript.segments, intervals, speaker_map=speaker_map_dict
        )
        # Diarization is authoritative — any speakers carried by the primary
        # provider are replaced. Real ASR providers don't populate speakers;
        # only the mock provider does, and there the diarizer's labels win.
        transcript = dc_replace(transcript, segments=new_segs, speakers=speakers)
        ensemble = dc_replace(ensemble, primary=transcript)

    has_post_stage = secondary_backend is not None or correct or diarize
    if has_post_stage:
        save_transcript(ensemble.primary, out / "transcript.raw.json")
    if ensemble.secondary is not None:
        save_transcript(ensemble.secondary, out / "transcript.secondary.json")

    if correct:
        corrector_obj = _build_corrector(
            corrector,
            correction_model,
            correction_host,
            correction_window,
            correction_overlap,
            title,
            participant_list,
        )
        transcript = corrector_obj.correct(ensemble.primary, ensemble.secondary_texts)

    save_transcript(transcript, out / "transcript.json")

    protocol = generate_protocol(transcript, title=title, participants=participant_list)
    (out / "protocol.md").write_text(render_protocol(protocol), encoding="utf-8")
    (out / "transcript.md").write_text(render_transcript(transcript), encoding="utf-8")
    (out / "actions.md").write_text(render_actions(protocol), encoding="utf-8")

    written = ["transcript.json"]
    if has_post_stage:
        written.append("transcript.raw.json")
    if ensemble.secondary is not None:
        written.append("transcript.secondary.json")
    written += ["protocol.md", "transcript.md", "actions.md"]
    typer.echo(f"Wrote {', '.join(written)} → {out}")


def _build_primary_backend(
    provider: str, model: str, whisper_cli: str, language: str, out: Path
) -> TranscriptionProvider:
    if provider == "mock":
        return MockTranscriptionProvider()
    if provider == "whisper-cpp":
        if not model:
            typer.echo("Error: --model is required for whisper-cpp provider.", err=True)
            raise typer.Exit(code=1)
        return WhisperCppProvider(
            model_path=Path(model),
            whisper_cli=whisper_cli,
            language=language,
            output_dir=out,
        )
    typer.echo(f"Error: unknown provider {provider!r}. Choose mock or whisper-cpp.", err=True)
    raise typer.Exit(code=1)


def _build_diarizer(
    name: str,
    hf_token: str,
    num_speakers: int,
    min_speakers: int,
    max_speakers: int,
    participants: list[str],
) -> Diarizer:
    if name != "pyannote":
        typer.echo(f"Error: unknown diarizer {name!r}. Choose pyannote.", err=True)
        raise typer.Exit(code=1)
    # 0 = unset for these flags. --num-speakers is an exact constraint in pyannote;
    # we never derive it from --participants (a 4-person team in a 1:1 would be
    # told there are 4 speakers, hallucinating two). --participants only contributes
    # a soft upper bound via --max-speakers when both that and --num-speakers are unset.
    num: int | None = num_speakers or None
    minimum: int | None = min_speakers or None
    maximum: int | None = max_speakers or None
    if num is None and maximum is None and participants:
        maximum = len(participants)
    try:
        return PyannoteDiarizer(
            auth_token=hf_token or None,
            num_speakers=num,
            min_speakers=minimum,
            max_speakers=maximum,
        )
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None


def _build_corrector(
    name: str,
    model: str,
    host: str,
    window: int,
    overlap: int,
    title: str,
    participants: list[str],
) -> OllamaCorrector:
    if name != "ollama":
        typer.echo(f"Error: unknown corrector {name!r}. Choose ollama.", err=True)
        raise typer.Exit(code=1)
    return OllamaCorrector(
        model=model,
        host=host,
        window=window,
        overlap=overlap,
        title=title,
        participants=participants,
        on_warning=lambda msg: typer.echo(f"warning: {msg}", err=True),
    )
