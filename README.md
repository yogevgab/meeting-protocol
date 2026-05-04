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
meeting-protocol transcribe recording.mp3 \
  --provider whisper-cpp \
  --model /path/to/ggml-medium.bin \
  --title "Weekly Sync" \
  --participants "Yogev,Tom" \
  --out ./outputs
```

This writes four files to `./outputs/`:

| File | Contents |
|---|---|
| `transcript.json` | Raw transcript in the internal JSON format |
| `protocol.md` | Full protocol with decisions, actions, ideas, open questions |
| `transcript.md` | Speaker-labelled transcript with timestamps |
| `actions.md` | Standalone action items checklist |

> **Current limitations:** Diarization is not yet implemented. The `mock` backend returns two fixed speakers (`S1=Yogev`, `S2=Tom`). The `whisper-cpp` backend produces segments with no speaker labels — all segments are left without a `speaker_id` until a diarization step is added.

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
│   └── diarization/       # Backend adapters (pluggable, not yet implemented)
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
