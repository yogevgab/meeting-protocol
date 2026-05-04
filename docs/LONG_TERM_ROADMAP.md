# Long-Term Roadmap: Open Meeting Intelligence Suite

> Working vision: build an open-source, local-first/open-core ecosystem for recording, transcribing, understanding, organizing, and acting on meetings, calls, chats, and voice interactions.

## Product Thesis

Modern work has too many fragmented communication channels: meetings, phone calls, WhatsApp, chatbots, voice agents, calendars, action items, and follow-ups. The long-term opportunity is to build a modular “communication intelligence layer” that captures conversations from many sources, turns them into reliable structured knowledge, and helps people execute.

The first project, `meeting-protocol`, is the local-first transcription/diarization/protocol foundation. The broader suite should be composed of small open-source projects with clean APIs that can later connect into a web app/platform.

## Guiding Principles

1. **Local-first where possible** — recordings and transcripts are sensitive.
2. **Open-source core** — each infrastructure component should be useful by itself.
3. **Composable modules** — every project exposes CLI/API/webhook interfaces.
4. **Human-verifiable output** — especially for summaries, tasks, and speaker identity.
5. **Hebrew + English first-class support** — not an afterthought.
6. **Provider-agnostic architecture** — support local models and cloud models.
7. **Privacy boundaries** — no accidental upload of private audio/transcripts.
8. **Quality gates** — typed code, tests, docs, smoke tests, no secret leaks.

---

## Current Project: meeting-protocol

**Role:** Local-first engine for meeting audio → transcript → speakers → protocol artifacts.

### Current Capabilities

- Accepts one or multiple audio files as one meeting.
- Auto-normalizes compressed audio inputs like M4A/MP3/AAC to temporary WAV.
- Transcribes Hebrew with whisper.cpp + Ivrit.ai model.
- Runs pyannote diarization.
- Produces transcript/protocol/action Markdown + JSON artifacts.
- Supports Ollama correction experiments.

### Known Weakness: Speaker Diarization

Diarization currently works technically, but quality is poor on real recordings: pyannote assigns most segments to one speaker and leaves many segments unassigned.

### Diarization Improvement Plan

#### Phase D1: Make diarization measurable

Goal: stop guessing; create an evaluation loop.

Tasks:
- Add optional `diarization.json` output with raw pyannote intervals.
- Add `speaker_assignment_debug.json` showing, per transcript segment:
  - candidate speaker intervals
  - overlap durations
  - selected speaker
  - confidence/coverage ratio
- Add a small manually-labeled fixture from non-private synthetic audio.
- Add CLI command or script to report:
  - % segments assigned
  - speaker distribution
  - unassigned segments
  - low-confidence assignments

#### Phase D2: Improve assignment logic

Goal: better map pyannote intervals to ASR segments.

Ideas:
- Assign by maximum overlap, not strict containment.
- Add minimum overlap ratio threshold.
- If segment has no overlap, check nearby intervals within tolerance window.
- Split long transcript segments when pyannote speaker changes inside them.
- Preserve `speaker_id=None` only when confidence is genuinely low.

#### Phase D3: Improve diarization audio input

Goal: give pyannote better audio.

Ideas:
- Normalize loudness before diarization.
- Use VAD/silence trimming carefully.
- Test pyannote 3.1 vs pyannote 4 pipeline defaults.
- Try `min_speakers=2 max_speakers=2` vs `num_speakers=2`.
- Benchmark on 1, 3, 5, and 10 minute slices.

#### Phase D4: Speaker identity / enrollment

Goal: map `SPEAKER_00` / `SPEAKER_01` to real people reliably.

Ideas:
- Add `--speaker-map` workflow with review UI/output.
- Add known-speaker voice samples: `--speaker-sample Yogev=yogev.wav`.
- Generate embeddings for known speakers and match clusters.
- Let user confirm mapping once per meeting/project.

#### Phase D5: Multi-channel future

Goal: better diarization when recording source allows separation.

Ideas:
- For meeting bots, record separate audio tracks per participant when platform/API allows it.
- For phone connectors, preserve call legs when available.
- Prefer channel/source identity over acoustic diarization when possible.

---

## Project 1: Meeting Recorder Bot

**Working name:** `meeting-recorder-bot`

**Purpose:** A bot/service that joins online meetings, records audio/video or audio-only, and sends recordings to `meeting-protocol`.

### Scope

Initial target should be **one meeting provider**, not all of them.

Recommended order:
1. Google Meet via browser automation/Chromium.
2. Zoom later.
3. Microsoft Teams later.

### MVP

- CLI/API to schedule a bot join:
  - meeting URL
  - title
  - start time / immediate join
  - output folder/webhook
- Bot joins meeting as a named participant.
- Records audio locally.
- Stores metadata:
  - meeting URL
  - start/end time
  - participants if visible
  - recording path
- Calls `meeting-protocol` after recording ends.

### Open-source value

A self-hosted meeting recording bot is useful even without the rest of the platform.

### Major risks

- Meeting platform ToS and consent requirements.
- Browser automation fragility.
- CAPTCHA/login flows.
- Capturing system/tab audio reliably on macOS/Linux.
- Headless vs headed browser differences.

### Architecture

- Python or Node service.
- Playwright for browser control.
- ffmpeg for capture/encoding.
- Job state stored in SQLite.
- Webhook callback on completion.

---

## Project 2: Meeting Summary / Protocol Generator

**Working name:** `meeting-intelligence`

**Purpose:** Take transcript + speakers + metadata and produce a professional meeting summary, decisions, action items, owners, due dates, risks, open questions, and follow-up messages.

### Why separate from `meeting-protocol`?

`meeting-protocol` should stay a reliable ingestion/transcription engine. Summarization is a separate intelligence layer with different model/prompt/evaluation concerns.

### Inputs

- `transcript.json` from `meeting-protocol`.
- Optional speaker map.
- Optional meeting title/context.
- Optional project/customer metadata.

### Outputs

- `summary.md`
- `decisions.md`
- `actions.md`
- `followup_email.md`
- `structured.json` with typed schema:
  - summary
  - topics
  - decisions
  - action_items
  - owners
  - due_dates
  - risks
  - open_questions
  - highlights

### MVP

- CLI:
  ```bash
  meeting-intelligence summarize transcript.json --out ./summary
  ```
- Local Ollama backend first.
- Cloud provider backend later.
- Strong JSON schema validation.
- Conservative mode: never invent action items not supported by transcript evidence.

### Evaluation

- Golden transcript fixtures.
- Snapshot tests for structure.
- Human rating workflow.
- Evidence links: every decision/action should cite transcript segment IDs.

---

## Project 3: Web App / Meeting OS

**Working name:** `meeting-os` or `workstream-os`

**Purpose:** Web application for managing meetings, recordings, transcripts, summaries, tasks, calendar context, and future communication modules.

### Product role

This becomes the user-facing hub. The other projects are engines/connectors.

### Core objects

- User
- Workspace
- Project
- Meeting
- Recording
- Transcript
- Speaker
- Summary
- Decision
- ActionItem
- Contact
- CalendarEvent
- Integration
- AgentRun

### MVP

- Upload or select local recording.
- Run transcription pipeline.
- View transcript with timestamps and speakers.
- Edit speaker names.
- Generate summary/action items.
- Manually edit/approve action items.
- Export Markdown/PDF/JSON.

### Later

- Calendar integration.
- Search across all meetings.
- Project-level memory.
- CRM-like contacts.
- WhatsApp/call/chatbot connectors.
- Team collaboration.
- Background jobs.
- Hosted SaaS option.

### Suggested stack

- Backend: FastAPI or Django.
- DB: Postgres.
- Jobs: Celery/RQ/Temporal-lite initially.
- Frontend: Next.js or React/Vite.
- Auth: Auth.js/Clerk/Supabase Auth initially, self-hostable alternative later.
- File storage: local filesystem first, S3-compatible later.

---

## Project 4: Phone Line Connector / Call Recorder

**Working name:** `phone-recorder-connector`

**Purpose:** Connect a phone number, record calls with consent, and send recordings into the meeting pipeline.

### MVP

- Twilio or similar provider first.
- Inbound call recording.
- Store recording + call metadata.
- Webhook to `meeting-protocol`.
- Basic call summary.

### Later

- Outbound calls.
- Call routing.
- Separate call legs if provider supports it.
- CRM/contact matching.
- Compliance/consent prompts.

### Risks

- Legal compliance differs by jurisdiction.
- Provider cost.
- Recording quality.
- Hebrew speech over phone is lower bandwidth; ASR quality may drop.

---

## Project 5: WhatsApp Read-Only Connector

**Working name:** `whatsapp-readonly-connector`

**Purpose:** Read WhatsApp messages into a private knowledge/task system without sending messages initially.

### MVP options

1. **Official WhatsApp Business Cloud API**
   - Reliable and compliant.
   - Best for business numbers.
   - Limited for personal WhatsApp history.

2. **Local export/import**
   - User exports chat.
   - CLI imports messages.
   - Good open-source MVP.

3. **WhatsApp Web automation**
   - Powerful but fragile and likely ToS-sensitive.
   - Avoid as first open-source path unless carefully scoped.

### MVP recommendation

Start with **chat export import**:

```bash
whatsapp-import chat.txt --contact "Tom" --out messages.json
```

Then later add official API support.

### Outputs

- Normalized message JSON.
- Contact mapping.
- Thread summaries.
- Extracted tasks/promises/questions.

---

## Project 6: Voice Agent Platform

**Working name:** `voice-agent-platform`

**Purpose:** Infrastructure for connecting a voice agent to phone/web/audio channels.

### Scope

This should come after transcription + summaries + phone connector are stable.

### Capabilities

- STT input.
- LLM reasoning/tool calling.
- TTS output.
- Call/session state.
- Interruptions/barge-in.
- Conversation logs.
- Escalation to human.

### MVP

- Web microphone demo.
- Local STT + local or cloud LLM + TTS.
- Simple tool: search meeting summaries / create action item.

### Later

- Phone agent via Twilio.
- Calendar-aware assistant.
- WhatsApp-aware assistant.
- Business automation agent.

---

## Project 7: Chatbot Platform

**Working name:** `chat-agent-platform`

**Purpose:** A text-agent framework connected to the same workspace data: meetings, summaries, tasks, contacts, calendar, WhatsApp, and calls.

### MVP

- Web chat UI.
- Agent can search meetings/transcripts/summaries.
- Agent can answer with citations.
- Agent can create/update tasks after user confirmation.

### Later

- Telegram/WhatsApp/Slack channels.
- Multi-agent workflows.
- Scheduled proactive summaries.
- Project-specific memory.

---

## Recommended Build Order

### Phase 0 — Stabilize current foundation

1. Improve diarization evaluation and assignment.
2. Add evidence-linked protocol generation.
3. Add robust long-audio chunking/progress/resume.
4. Add a documented stable JSON schema.

### Phase 1 — Meeting Intelligence

Build `meeting-intelligence` next.

Reason: it creates immediate value from transcripts we already produce, and it is independent of browser/meeting-platform complexity.

Deliverable:
- transcript → professional summary/actions/decisions/follow-up.

### Phase 2 — Web App MVP

Build `meeting-os` as the hub.

Reason: once transcription + summarization work, a UI becomes valuable.

Deliverable:
- upload recording
- run pipeline
- view/edit transcript
- approve action items
- search meetings

### Phase 3 — Recorder Bot

Build `meeting-recorder-bot`.

Reason: after the web app exists, automatic recording has a natural destination.

Deliverable:
- join Google Meet
- record
- push into Meeting OS

### Phase 4 — Communication Connectors

Build connectors in this order:
1. WhatsApp export importer
2. Phone recorder connector
3. Official WhatsApp API connector

### Phase 5 — Agents

Build agent platforms after data layer exists:
1. Chatbot platform
2. Voice agent platform

Reason: agents need trusted data and tools. Without the meeting/task/contact substrate, they are demos rather than products.

---

## Monorepo vs Multi-Repo

Recommended: **multi-repo with shared schemas**.

Repos:
- `meeting-protocol` — audio/transcription/diarization engine.
- `meeting-intelligence` — transcript → summary/tasks/decisions.
- `meeting-recorder-bot` — meeting join/record service.
- `meeting-os` — web app and API.
- `phone-recorder-connector` — phone integration.
- `whatsapp-readonly-connector` — WhatsApp import/read connectors.
- `agent-runtime` — shared chat/voice agent runtime later.
- `meeting-schemas` — optional shared schema package once duplication hurts.

Start with duplicated simple JSON schemas if needed; extract `meeting-schemas` only when two or more repos truly need it.

---

## Business / Portfolio Strategy

This suite can support multiple paths:

1. **Open-source credibility** — each repo is useful by itself.
2. **Freelance offering** — custom meeting/call/WhatsApp automation for small businesses.
3. **SaaS** — hosted Meeting OS for teams.
4. **Content** — document building the business publicly with Tom/Yogev.
5. **Consulting wedge** — “we turn your meetings/calls/messages into an execution system.”

Potential initial niche:
- Hebrew-speaking freelancers/small businesses.
- Lawyers/consultants/real-estate/clinics/coaches.
- Anyone with high-value meetings and poor follow-up tracking.

---

## Immediate Next Actions

1. Create `meeting-protocol` issue: diarization evaluation/debug outputs.
2. Create `meeting-protocol` issue: improve assignment overlap logic.
3. Start `meeting-intelligence` repo with strict schema-first CLI.
4. Define shared `Transcript` and `MeetingSummary` JSON schemas.
5. Run a real meeting through current pipeline and manually label 20–50 segments for diarization evaluation.
6. Decide name/positioning for the suite.

## Current Recommendation

Do **not** start all seven projects at once.

Best sequence:

1. Make `meeting-protocol` reliable enough for repeated use.
2. Build `meeting-intelligence` because it produces the professional output users actually want.
3. Build a minimal web app around those two.
4. Only then add bots/connectors/agents.

This keeps momentum high while still building toward the full platform vision.
