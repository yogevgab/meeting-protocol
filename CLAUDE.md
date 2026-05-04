# Claude Instructions for Meeting Protocol

You are building an open-source-quality local-first toolkit for meeting transcription, speaker identification/diarization, and protocol/minutes generation, optimized for Hebrew + English mixed conversations.

## Constraints
- Prefer Python unless there is a strong reason otherwise.
- Use modern packaging and typed code.
- Keep core functionality local-first and privacy-aware.
- Do not add heavyweight dependencies unless optional extras are appropriate.
- Build a useful skeleton first with tests and docs; real transcription backend can be adapter-based.
- This repo should be publishable as open source.

## Quality bar
- Clear README with installation, quickstart, roadmap, architecture.
- MIT license already chosen via GitHub.
- Tests must pass locally.
- Add GitHub Actions CI.
- Use ruff/mypy/pytest or equivalent quality gates.
- Include sample Hebrew/English fixture data.
- Keep the project modular so future whisper.cpp/faster-whisper/pyannote integrations are easy.

## User context
This project is for Yogev + Tom's shared business/social documentation pipeline. The first real production use is transcribing recorded business-building meetings on a Mac mini and generating Obsidian-friendly Markdown protocols.
