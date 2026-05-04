from meeting_protocol.models import ActionStatus, Protocol, Transcript


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_protocol(protocol: Protocol) -> str:
    lines: list[str] = [
        f"# {protocol.title}",
        "",
        f"**Date:** {protocol.date.strftime('%Y-%m-%d')}",
        f"**Participants:** {', '.join(protocol.participants)}",
        f"**Duration:** {_fmt_duration(protocol.duration)}",
        f"**Status:** {protocol.status}",
        "",
    ]
    if protocol.tldr:
        lines += ["## tl;dr", "", protocol.tldr, ""]
    if protocol.decisions:
        lines += ["## Decisions", ""] + [f"- {d.description}" for d in protocol.decisions] + [""]
    if protocol.action_items:
        lines.append("## Action Items")
        lines.append("")
        for a in protocol.action_items:
            box = "x" if a.status == ActionStatus.DONE else " "
            owner = f" — {a.owner}" if a.owner else ""
            lines.append(f"- [{box}] {a.description}{owner}")
        lines.append("")
    if protocol.open_questions:
        lines += ["## Open Questions", ""] + [f"- {q}" for q in protocol.open_questions] + [""]
    if protocol.business_ideas:
        lines += ["## Business Ideas", ""] + [f"- {h.text}" for h in protocol.business_ideas] + [""]
    if protocol.content_opportunities:
        lines += (
            ["## Content Opportunities", ""]
            + [f"- {h.text}" for h in protocol.content_opportunities]
            + [""]
        )
    return "\n".join(lines)


def render_transcript(transcript: Transcript) -> str:
    speaker_map = {sp.id: sp.name for sp in transcript.speakers}
    lines: list[str] = [
        "# Transcript",
        "",
        f"**Source:** {transcript.source_file}",
        f"**Duration:** {_fmt_duration(transcript.duration)}",
    ]
    if transcript.recorded_at:
        lines.append(f"**Recorded:** {transcript.recorded_at.strftime('%Y-%m-%d %H:%M')}")
    lines += ["", "## Segments", ""]
    for seg in transcript.segments:
        sid = seg.speaker_id
        name = speaker_map.get(sid, sid) if sid is not None else "Unknown"
        lang = f" ({seg.language})" if seg.language else ""
        ts = f"[{_fmt_ts(seg.start)} → {_fmt_ts(seg.end)}]"
        lines.append(f"**{ts} {name}{lang}:** {seg.text}")
    lines.append("")
    return "\n".join(lines)


def render_actions(protocol: Protocol) -> str:
    lines: list[str] = [f"# Action Items: {protocol.title}", ""]
    if not protocol.action_items:
        lines.append("*No action items.*")
    else:
        for a in protocol.action_items:
            box = "x" if a.status == ActionStatus.DONE else " "
            owner = f" — {a.owner}" if a.owner else ""
            due = f" *(due: {a.due_date})*" if a.due_date else ""
            lines.append(f"- [{box}] {a.description}{owner}{due}")
    lines.append("")
    return "\n".join(lines)
