"""Activity engine — manages interactive activities like trivia, stories, etc."""

import yaml
import os

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activities")

with open(os.path.join(_DIR, "activities.yaml")) as f:
    _activities = yaml.safe_load(f)["activities"]

# Current activity state
_active = None  # e.g. {"type": "trivia", "context": "...injected prompt..."}


def get_activity_context() -> str:
    """Return current activity context for injection into system prompt."""
    if _active:
        return _active["context"]
    return ""


def check_trigger(user_text: str) -> str | None:
    """Check if user text triggers an activity. Returns a response or None."""
    text = user_text.lower().strip()

    # Check for stop/done
    if _active:
        for word in ["stop", "done", "quit", "enough", "no more", "i'm done", "that's enough"]:
            if word in text:
                _clear()
                return None
        return None  # Activity is running, let LLM handle it with context

    # Check activity triggers
    for act in _activities:
        if any(t in text for t in act["trigger"]):
            if act["name"] == "Trivia":
                return _start_trivia_menu(act)
            # Other activities don't need special context
            return None

    # Check if picking a trivia category/topic (when no activity active but browsing)
    return None


def check_trivia_selection(user_text: str) -> str | None:
    """Check if user is selecting a trivia category or topic."""
    global _active
    text = user_text.lower().strip()

    if _active and _active.get("selecting"):
        level = _active["selecting"]

        if level == "category":
            for cat in _active["categories"]:
                if any(t in text for t in cat["trigger"]):
                    if len(cat["topics"]) == 1:
                        return _load_topic(cat["topics"][0])
                    _active["selecting"] = "topic"
                    _active["topics"] = cat["topics"]
                    names = [t["name"] for t in cat["topics"]]
                    return f"I have {', '.join(names[:-1])}, or {names[-1]}. Which one?"

        elif level == "topic":
            for topic in _active["topics"]:
                if any(t in text for t in topic["trigger"]):
                    return _load_topic(topic)
            # If no match, try first word match
            for topic in _active["topics"]:
                if topic["name"].lower().split()[0] in text:
                    return _load_topic(topic)

    return None


def is_selecting() -> bool:
    """True if we're in a trivia selection menu."""
    return _active is not None and _active.get("selecting") is not None


def is_active() -> bool:
    """True if an activity is currently running."""
    return _active is not None and not _active.get("selecting")


def _start_trivia_menu(act) -> str:
    global _active
    cats = act["categories"]
    _active = {"type": "trivia", "context": "", "selecting": "category", "categories": cats}
    names = [c["name"] for c in cats]
    return f"I can quiz you on {', '.join(names[:-1])}, or {names[-1]}. What sounds fun?"


def _load_topic(topic) -> str:
    global _active
    filepath = os.path.join(_DIR, topic["file"])
    with open(filepath) as f:
        data = yaml.safe_load(f)

    import random
    questions = data["questions"][:]
    random.shuffle(questions)

    # Build context block with questions and answers
    qa_lines = []
    for q in questions:
        qa_lines.append(f"Q: {q['q']}")
        qa_lines.append(f"A: {q['a']}")
        qa_lines.append(f"Hint: {q['hint']}")
        qa_lines.append("")

    context = f"""ACTIVITY MODE: Trivia — {data['topic']}
{data['instructions']}

Use ONLY these questions and answers. Do NOT make up your own.
Shuffle the order. Track which ones you've asked.

{chr(10).join(qa_lines)}

When the user says they're done, say something encouraging and end the quiz."""

    _active = {"type": "trivia", "context": context}
    return f"Great choice! Let's do {data['topic']} trivia."


def _clear():
    global _active
    _active = None
