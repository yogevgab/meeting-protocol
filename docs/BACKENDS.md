# Backend Adapters

meeting-protocol uses an adapter pattern for transcription and diarization so the heavy ML dependencies remain optional. The core package (`typer`, `rich`) stays lightweight.

---

## Transcription Backends

| Backend | Status | Install extra | Notes |
|---|---|---|---|
| `MockTranscriptionProvider` | **Implemented** | *(none)* | Returns fixed fixture segments; for tests and offline dev |
| `WhisperCppProvider` | **Implemented** | *(none — whisper-cli must be on PATH)* | Shells out to `whisper-cli`; parses JSON output |
| `FasterWhisperProvider` | Planned | `.[faster-whisper]` | Uses `faster-whisper` Python package; GPU-optional |

### MockTranscriptionProvider

Returns three hardcoded segments with two speakers (`S1=Yogev`, `S2=Tom`). No audio processing is performed — the audio path is recorded as `source_file` but the file is not read.

Use `--provider mock` on the CLI:

```bash
meeting-protocol transcribe recording.mp3 --provider mock --out ./outputs
```

### WhisperCppProvider

Shells out to [`whisper-cli`](https://github.com/ggerganov/whisper.cpp) (the renamed `main` binary in whisper.cpp ≥ 1.7). The provider passes `--output-json` and `--output-file <out>/<stem>` so whisper-cli writes a JSON sidecar (`<out>/<stem>.json`) into the output directory. If stdout is empty (the common case on current builds), the provider reads the sidecar directly; if stdout contains JSON, it uses that instead.

**Expected JSON format from `whisper-cli --output-json`:**

```json
{
  "transcription": [
    {
      "offsets": { "from": 0, "to": 5000 },
      "text": " Hello, let's begin."
    }
  ]
}
```

- `offsets.from` and `offsets.to` are in **milliseconds**; the provider converts them to seconds.
- No `speaker_id` is assigned — speaker diarization is not yet implemented.

Use `--provider whisper-cpp` with a required `--model` path:

```bash
meeting-protocol transcribe recording.mp3 \
  --provider whisper-cpp \
  --model /path/to/ggml-medium.bin \
  --out ./outputs
```

The binary name defaults to `whisper-cli`. Override it with `--whisper-cli` on the CLI or with the `whisper_cli` constructor argument when using the provider programmatically:

```bash
# Override binary name on the CLI
meeting-protocol transcribe recording.mp3 \
  --provider whisper-cpp \
  --model /path/to/ggml-medium.bin \
  --whisper-cli /usr/local/bin/whisper-cli \
  --out ./outputs
```

```python
from pathlib import Path
from meeting_protocol.transcription.whisper_cpp import WhisperCppProvider

provider = WhisperCppProvider(
    model_path=Path("/path/to/ggml-medium.bin"),
    whisper_cli="whisper-main",  # override if binary is named differently
)
transcript = provider.transcribe(Path("recording.mp3"))
```

**macOS / Homebrew:** `whisper-cli` is available via `brew install whisper-cpp`. Homebrew may also install an empty `for-tests` tiny model at a path like `/opt/homebrew/Cellar/whisper-cpp/<version>/share/whisper-cpp/for-tests-ggml-tiny.bin`; this file is intentionally minimal and **not suitable for real transcription** — use it only to verify the binary works end-to-end.

**Current limitations:**

- Speaker diarization is **not implemented**. All output segments have no `speaker_id`.

### Implementing a custom transcription backend

```python
from pathlib import Path
from meeting_protocol.models import Transcript
from meeting_protocol.transcription.base import TranscriptionProvider

class MyTranscriptionProvider(TranscriptionProvider):
    def transcribe(self, audio_path: Path) -> Transcript:
        ...
```

The provider must return a fully populated `Transcript` (segments with `id`, `start`, `end`, `text`). Speaker IDs are optional at this stage — a diarization provider can assign them afterwards.

---

## Diarization Backends

| Backend | Status | Install extra | Notes |
|---|---|---|---|
| `ManualDiarizationProvider` | Planned | *(none)* | Accepts a mapping of `{speaker_id: name}` provided by the user |
| `PyannoteProvider` | Planned | `.[pyannote]` | Uses `pyannote.audio` for automatic speaker diarization |

> **Note:** Diarization is not yet wired into the CLI. The `transcribe` command currently produces transcripts with no speaker labels when using the `whisper-cpp` backend.

### Implementing a custom diarization backend

```python
from meeting_protocol.models import Transcript

class MyDiarizationProvider:
    def diarize(self, transcript: Transcript) -> Transcript:
        # Assign segment.speaker_id values, return updated transcript
        ...
```

---

## CLI — `transcribe` command

The `transcribe` command is live. It accepts `--provider` and `--model`:

```bash
meeting-protocol transcribe recording.mp3 \
  --provider whisper-cpp \
  --model /path/to/ggml-medium.bin \
  --title "Weekly Sync" \
  --participants "Yogev,Tom" \
  --out ./outputs
```

| Option | Default | Description |
|---|---|---|
| `--provider` | `mock` | Backend to use: `mock` or `whisper-cpp` |
| `--model` | *(empty)* | Path to whisper.cpp `.bin` model file (required for `whisper-cpp`) |
| `--whisper-cli` | `whisper-cli` | Name or path of the whisper-cli binary (`whisper-cpp` only) |
| `--title` / `-t` | `"Meeting Protocol"` | Protocol title |
| `--participants` / `-p` | *(empty)* | Comma-separated participant names |
| `--out` / `-o` | `.` | Output directory (created if absent) |

Outputs written: `transcript.json`, `protocol.md`, `transcript.md`, `actions.md`.

Future backends (`faster-whisper`, diarizers) will be added as additional `--provider` values without breaking existing calls.

---

## Priority order

1. `MockTranscriptionProvider` — ✅ implemented; unblocks full integration tests without hardware
2. `WhisperCppProvider` — ✅ implemented; shells out to `whisper-cli`, no speaker diarization yet
3. `ManualDiarizationProvider` — enables speaker labelling from existing transcripts
4. `FasterWhisperProvider` — first real Python transcription backend (Mac mini target)
5. `PyannoteProvider` — automatic diarization (requires GPU or patience)
