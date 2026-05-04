# Project Brief: Meeting Protocol

## Vision
Build an open-source, local-first toolkit that turns recorded meetings into high-quality searchable transcripts, speaker-attributed notes, and structured protocols/minutes. It should work especially well for Hebrew + English mixed conversations on a Mac mini, while remaining cross-platform when possible.

## Primary use case
Yogev and Tom record their business-building meetings. The tool should:
1. Accept audio/video files.
2. Transcribe accurately in Hebrew, English, and mixed Hebrew/English.
3. Identify different participants/speakers.
4. Generate a structured meeting protocol in Markdown.
5. Extract decisions, action items, business ideas, quotes/highlights, and short-form content opportunities.
6. Store outputs in a clean folder structure that can be used with Obsidian/Git.

## Product principles
- Open-source quality: clear README, license, tests, CI, typed code, developer docs.
- Local-first: no cloud dependency for core transcription/diarization.
- Practical CLI first; library API second.
- Extensible: adapters for whisper.cpp, faster-whisper, diarization engines, and LLM summarizers.
- Privacy-aware: recordings and transcripts stay local by default.
- Good Hebrew support: document model recommendations and tradeoffs for Hebrew/English.

## Suggested technical direction
- Python project with modern packaging (`pyproject.toml`).
- CLI using Typer or Click.
- Transcription provider abstraction:
  - `whisper.cpp` CLI adapter as first target.
  - `faster-whisper` optional adapter later.
- Diarization provider abstraction:
  - Start with transcript segment speaker placeholders / manual speaker mapping.
  - Optional future integration with pyannote.audio or whisperX.
- Outputs:
  - `transcript.md`
  - `transcript.json`
  - `transcript.srt` or `transcript.vtt`
  - `protocol.md`
  - `actions.md`
  - `highlights.md`
- Protocol format should include:
  - title/date/participants/source file/duration/status
  - TL;DR
  - agenda/topics
  - decisions
  - action items with owner/due/status
  - risks/open questions
  - business ideas
  - content opportunities
  - full transcript link/section

## Initial milestone
Create a working repository skeleton that can run locally and in CI, with:
- CLI command(s)
- typed domain models
- mock transcription engine for tests
- protocol generator from transcript JSON
- sample fixture transcript in Hebrew/English
- tests for protocol/action/highlight extraction heuristics
- docs explaining how to later install real whisper.cpp/faster-whisper backends

## Non-goals for first commit
- Do not require GPU/cloud.
- Do not require perfect speaker diarization yet.
- Do not build a GUI yet.
- Do not depend on paid APIs for the core path.
