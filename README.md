# meeting-protocol

Local-first toolkit for meeting transcription, speaker diarization, and protocol/minutes generation, optimized for Hebrew + English mixed conversations.

Built for Obsidian-friendly Markdown output. Privacy-first: all processing runs on your machine.

---

## Features

- Load JSON transcripts and generate structured meeting protocols
- Heuristic extraction of decisions, action items, business ideas, content opportunities, and open questions — in both Hebrew and English
- Renders Obsidian-compatible Markdown: `protocol.md`, `transcript.md`, `actions.md`
- Adapter-based transcription and diarization backends (plug in whisper.cpp, faster-whisper, pyannote)
- Typed Python codebase with strict mypy, ruff, and pytest

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/yogevgab/meeting-protocol
cd meeting-protocol
pip install -e ".[dev]"
```

For faster-whisper transcription support:

```bash
pip install -e ".[faster-whisper]"
```

---

## Quickstart

### Transcribe audio and generate a protocol

Use the `transcribe` command to go straight from an audio file to a full protocol:

```bash
# Mock backend (no hardware required — useful for testing and CI)
meeting-protocol transcribe recording.mp3 \
  --provider mock \
  --title "Weekly Sync" \
  --participants "Yogev,Tom" \
  --out ./outputs

# whisper-cpp backend (requires whisper-cli on PATH and a model file)
# On macOS: brew install whisper-cpp  →  whisper-cli is placed on PATH
meeting-protocol transcribe recording.mp3 \
  --provider whisper-cpp \
  --model /path/to/ggml-medium.bin \
  --title "Weekly Sync" \
  --participants "Yogev,Tom" \
  --out ./outputs

# If your binary is named differently or lives at a custom path, pass --whisper-cli:
meeting-protocol transcribe recording.mp3 \
  --provider whisper-cpp \
  --model /path/to/ggml-medium.bin \
  --whisper-cli /usr/local/bin/whisper-cli \
  --out ./outputs
```

### Transcribing a meeting split across multiple audio files

If a meeting was recorded in parts (e.g. the recorder was paused and restarted, or the session spanned multiple files), pass all files to `transcribe` in one command:

```bash
meeting-protocol transcribe part1.m4a part2.m4a part3.m4a \
  --provider whisper-cpp \
  --model ~/.cache/whisper-models/ivrit-large-v3-turbo/ggml-model.bin \
  --language he \
  --title "Weekly Sync" \
  --participants "Yogev,Tom" \
  --out ./outputs
```

Files are processed in the order given. Timestamps from each subsequent file are offset so the full timeline is continuous — the result is a single merged transcript as if it were one recording. The usual four output files (`transcript.json`, `protocol.md`, `transcript.md`, `actions.md`) are produced for the combined session.

**Current limitation — diarization with multiple files is not yet supported.** If you pass `--diarize` together with multiple audio files the command will exit with an error. Workaround: concatenate the files into one before running:

```bash
ffmpeg -i "concat:part1.m4a|part2.m4a|part3.m4a" -c copy combined.m4a
meeting-protocol transcribe combined.m4a --diarize ...
```

Or convert each part to WAV and join them:

```bash
for f in part1.m4a part2.m4a part3.m4a; do
  ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.m4a}.wav"
done
sox part1.wav part2.wav part3.wav combined.wav
meeting-protocol transcribe combined.wav --diarize ...
```

### Hebrew transcription (Ivrit model)

For Hebrew or Hebrew/English mixed recordings, use the **ivrit-ai/whisper-large-v3-turbo-ggml** model and pass `--language he`. The `--language` flag defaults to `auto` (whisper.cpp language detection), but language detection can mis-classify Hebrew — always specify it explicitly for Hebrew recordings.

**Download the model once to a local cache directory** (models are large; do not store them inside the repo):

```bash
pip install huggingface-hub   # one-time
hf download ivrit-ai/whisper-large-v3-turbo-ggml \
  --local-dir ~/.cache/whisper-models/ivrit-large-v3-turbo
```

**Transcribe a Hebrew meeting:** whisper.cpp supports WAV/MP3/FLAC/OGG. If your recorder produces M4A, convert it first with `ffmpeg -i recording.m4a -ar 16000 -ac 1 recording.wav`.

```bash
meeting-protocol transcribe recording.wav \
  --provider whisper-cpp \
  --model ~/.cache/whisper-models/ivrit-large-v3-turbo/ggml-model.bin \
  --language he \
  --title "פגישה שבועית" \
  --participants "Yogev,Tom" \
  --out ./outputs
```

**Observed local smoke test (macOS, whisper-cli 1.8.3):** 6.8 s of synthetic Hebrew speech (`say -v Carmit`) was transcribed to Hebrew in ~2–3 seconds using the Ivrit large-v3-turbo model. The transcript was correct; a minor TTS artefact (the voice rendered עסק as *אִיסָק*) appeared faithfully in the output — this is a synthetic-audio limitation, not a model bug. Real meeting recordings still need evaluation.

> See [docs/BACKENDS.md](docs/BACKENDS.md) for full backend details, model download instructions, and the `--language` option reference.

This writes four files to `./outputs/`:

| File | Contents |
|---|---|
| `transcript.json` | Raw transcript in the internal JSON format |
| `protocol.md` | Full protocol with decisions, actions, ideas, open questions |
| `transcript.md` | Speaker-labelled transcript with timestamps |
| `actions.md` | Standalone action items checklist |

### Speaker diarization (optional)

The `--diarize` flag adds a "who spoke when" stage that assigns a `speaker_id` to each segment. The current backend is `pyannote.audio` (the academic gold standard, runs locally on Apple Silicon via PyTorch MPS).

**One-time setup** (HuggingFace requires you to accept two model licenses and provide an auth token):

1. Accept the license at <https://huggingface.co/pyannote/speaker-diarization-3.1>
2. Accept the license at <https://huggingface.co/pyannote/segmentation-3.0>
3. Create a read token at <https://huggingface.co/settings/tokens>
4. Install the optional dep and export the token:
   ```bash
   pip install -e .[pyannote]
   export HF_TOKEN=<your-token>
   ```

**Run with diarization** (and optional rename of anonymous labels):

```bash
meeting-protocol transcribe ./recording.wav \
  --provider whisper-cpp \
  --whisper-cli /opt/homebrew/bin/whisper-cli \
  --model /path/to/ggml-ivrit-large-v3-turbo.bin \
  --language he \
  --diarize \
  --num-speakers 2 \
  --speaker-map "SPEAKER_00=Yogev,SPEAKER_01=Tom" \
  --participants "Yogev,Tom" \
  --out ./outputs
```

After a first run, inspect `transcript.md` to see which anonymous label corresponds to which voice, then re-run with `--speaker-map` to put real names on them.

When combined with `--correct`, the LLM corrector receives a `"speaker"` field per segment as read-only context — useful for proper-name disambiguation.

> **Notes.** `--num-speakers` is an *exact* count when set; if you don't know it, leave at `0` (auto) or use `--max-speakers N` as a soft hint. Overlapped speech is collapsed to one dominant speaker per ASR segment.

### LLM correction with Ollama (optional)

Use `--correct` to run a conservative local LLM pass that fixes clear ASR mistakes without summarizing or rewriting the speaker's wording.

The code default is `gemma3:12b`, but on Yogev's Mac the currently installed general-purpose model is `qwen3:14b`. For local Hebrew correction, start with:

```bash
meeting-protocol transcribe recording.wav \
  --provider whisper-cpp \
  --model /path/to/ggml-ivrit-large-v3-turbo.bin \
  --language he \
  --correct \
  --correction-model qwen3:14b \
  --out outputs
```

Good comparison candidates are `qwen3:14b`, `gemma4:latest`, and larger Gemma/Qwen models if your Mac has enough free memory. For thinking models such as `gemma4:26b`, disable extended thinking during conservative ASR correction:

```bash
meeting-protocol transcribe recording.wav \
  --provider whisper-cpp \
  --model /path/to/ggml-ivrit-large-v3-turbo.bin \
  --language he \
  --correct \
  --correction-model gemma4:26b \
  --correction-think false \
  --out outputs
```

Always inspect `transcript.raw.json` and `transcript.json` side by side: correction should fix ASR errors, not paraphrase the meeting.

### Generate a protocol from an existing JSON transcript

If you already have a transcript in the internal format:

```bash
meeting-protocol from-transcript recording.json \
  --title "Weekly Sync" \
  --participants "Yogev,Tom" \
  --out ./outputs
```

This writes `protocol.md`, `transcript.md`, and `actions.md` (no `transcript.json` — the source file is used as-is).

### Check the installed version

```bash
meeting-protocol version
```

### Transcript JSON format

See [`fixtures/sample_transcript.json`](fixtures/sample_transcript.json) for a complete example. The minimal shape:

```json
{
  "segments": [
    { "id": 1, "start": 0.0, "end": 8.5, "text": "Let's begin.", "speaker_id": "S1", "language": "en" }
  ],
  "duration": 8.5,
  "source_file": "recording.mp4",
  "speakers": [{ "id": "S1", "name": "Yogev" }]
}
```

---

## Architecture

```
meeting-protocol/
├── src/meeting_protocol/
│   ├── models.py          # Domain models: Transcript, Protocol, Segment, …
│   ├── io.py              # JSON load/save for transcripts
│   ├── protocol/
│   │   └── generator.py   # Heuristic extractor (Hebrew + English regex)
│   ├── outputs/
│   │   └── markdown.py    # Markdown renderers: render_protocol, render_transcript, render_actions
│   ├── cli/
│   │   └── main.py        # Typer CLI: version, from-transcript, transcribe
│   ├── transcription/
│   │   ├── base.py        # TranscriptionProvider ABC
│   │   ├── mock.py        # MockTranscriptionProvider — fixed fixture segments, no hardware
│   │   └── whisper_cpp.py # WhisperCppProvider — shells out to whisper-cli, parses JSON output
│   ├── diarization/       # Diarizer ABC + PyannoteDiarizer + assignment helpers
│   └── correction/        # TranscriptCorrector ABC + OllamaCorrector (LLM post-correction)
├── fixtures/
│   └── sample_transcript.json
└── tests/
```

**Data flow:**

```
Audio file → TranscriptionProvider.transcribe() → Transcript → save_transcript() → transcript.json
                                                       ↓
JSON file  → load_transcript()              → Transcript
                                                       ↓
                                              generate_protocol()
                                                       ↓
                                                   Protocol
                                                       ↓
                   render_protocol() / render_transcript() / render_actions()
                                                       ↓
                                          Obsidian Markdown files
```

See [docs/BACKENDS.md](docs/BACKENDS.md) for the transcription and diarization backend roadmap.

---

## Development

```bash
# Run tests
pytest -v

# Lint
ruff check src tests

# Type check
mypy src
```

---

## License

MIT
