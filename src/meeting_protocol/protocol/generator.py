import re
from datetime import datetime

from meeting_protocol.models import ActionItem, Decision, Highlight, Protocol, Transcript

_DECISION_EN = re.compile(r"\b(decided|decision|we agreed|resolved|agreed to)\b", re.I)
_DECISION_HE = re.compile(r"(הוחלט|נחליט|החלטנו|הסכמנו)")
_ACTION_EN = re.compile(
    r"\b(i'll|i will|we will|we should|follow up|action item|todo|need to)\b", re.I
)
_ACTION_HE = re.compile(r"(צריך|יש ל|אנחנו צריכים|אני אצור|אני אעשה)")
_IDEA_EN = re.compile(r"\b(business idea|what if we|maybe we should)\b", re.I)
_IDEA_HE = re.compile(r"(רעיון עסקי|אפשר ל|מה אם נ)")
_CONTENT_EN = re.compile(
    r"\b(content opportunity|we could write|write about|post about)\b", re.I
)
_CONTENT_HE = re.compile(r"(פוסט על|תוכן על|לכתוב על)")
_QUESTION_EN = re.compile(
    r"\b(open question|need to decide|tbd|to be decided|not sure)\b", re.I
)
_QUESTION_HE = re.compile(r"(שאלה פתוחה|צריך להחליט|לא בטוח|לא ברור)")


def generate_protocol(
    transcript: Transcript,
    title: str,
    participants: list[str],
) -> Protocol:
    decisions: list[Decision] = []
    action_items: list[ActionItem] = []
    business_ideas: list[Highlight] = []
    content_opportunities: list[Highlight] = []
    open_questions: list[str] = []

    for seg in transcript.segments:
        text = seg.text
        if _DECISION_EN.search(text) or _DECISION_HE.search(text):
            decisions.append(Decision(description=text.strip(), segment_id=seg.id))
        if _ACTION_EN.search(text) or _ACTION_HE.search(text):
            action_items.append(ActionItem(description=text.strip(), segment_id=seg.id))
        if _IDEA_EN.search(text) or _IDEA_HE.search(text):
            business_ideas.append(
                Highlight(text=text.strip(), highlight_type="idea", segment_id=seg.id)
            )
        if _CONTENT_EN.search(text) or _CONTENT_HE.search(text):
            content_opportunities.append(
                Highlight(
                    text=text.strip(),
                    highlight_type="content_opportunity",
                    segment_id=seg.id,
                )
            )
        if _QUESTION_EN.search(text) or _QUESTION_HE.search(text):
            open_questions.append(text.strip())

    return Protocol(
        title=title,
        date=transcript.recorded_at or datetime.now(),
        participants=participants,
        source_file=transcript.source_file,
        duration=transcript.duration,
        decisions=decisions,
        action_items=action_items,
        business_ideas=business_ideas,
        content_opportunities=content_opportunities,
        open_questions=open_questions,
        transcript=transcript,
    )
