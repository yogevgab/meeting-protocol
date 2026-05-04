# Backend Adapters

meeting-protocol uses an adapter pattern for transcription and diarization so the heavy ML dependencies remain optional. The core package (`typer`, `rich`) stays lightweight.

---

## Transcription Backends

| Backend | Status | Install extra | Notes |
|---|---|---|---|
| `MockTranscriptionProvider` | Planned | *(none)* | Returns fixture data; for tests and offline dev |
| `WhisperCppProvider` | Planned | `.[whisper-cpp]` | Shells out to the `whisper-cpp` binary; no Python binding required |
| `FasterWhisperProvider` | Planned | `.[faster-whisper]` | Uses `faster-whisper` Python package; GPU-optional |

### Implementing a custom transcription backend

```python
from pathlib import Path
from meeting_protocol.models import Transcript

class MyTranscriptionProvider:
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

### Implementing a custom diarization backend

```python
from meeting_protocol.models import Transcript

class MyDiarizationProvider:
    def diarize(self, transcript: Transcript) -> Transcript:
        # Assign segment.speaker_id values, return updated transcript
        ...
```

---

## Planned CLI integration

Once backends are implemented, the CLI will accept `--backend` and `--diarizer` flags:

```bash
meeting-protocol transcribe recording.mp3 \
  --backend faster-whisper \
  --diarizer manual \
  --speakers "S1=Yogev,S2=Tom" \
  --out ./outputs
```

---

## Priority order

1. `MockTranscriptionProvider` — unblocks full integration tests without hardware
2. `ManualDiarizationProvider` — enables speaker labelling from existing transcripts
3. `FasterWhisperProvider` — first real transcription backend (Mac mini target)
4. `WhisperCppProvider` — alternative for lower-memory environments
5. `PyannoteProvider` — automatic diarization (requires GPU or patience)
