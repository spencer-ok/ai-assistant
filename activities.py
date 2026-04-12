"""Activity engine — manages interactive activities like trivia, word games, etc."""

import yaml
import os
import random

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activities")

with open(os.path.join(_DIR, "activities.yaml")) as f:
    _cfg = yaml.safe_load(f)

_TRIGGERS = _cfg["triggers"]
_activities = [a for a in _cfg["activities"] if a.get("enabled", True)]

# Current activity state
_active = None


def get_activity_context() -> str:
    if _active and _active.get("context"):
        return _active["context"]
    return ""


def check_trigger(user_text: str) -> str | None:
    text = user_text.lower().strip()

    # Check for stop/done
    if _active and not _active.get("selecting"):
        for word in ["stop", "done", "quit", "enough", "no more", "i'm done", "that's enough"]:
            if word in text:
                _clear()
                return "done"
        return None

    # If already selecting, handle selection
    if _active and _active.get("selecting"):
        return _handle_selection(text)

    # Check if user wants an activity
    if any(t in text for t in _TRIGGERS):
        return _show_menu()

    return None


def _show_menu() -> str:
    global _active
    names = [a["name"] for a in _activities]
    _active = {"selecting": "activity"}
    listing = ", ".join(names[:-1]) + ", or " + names[-1]
    return f"I can do {listing}. What sounds fun?"


def _handle_selection(text: str) -> str | None:
    global _active
    level = _active.get("selecting")

    if level == "activity":
        for act in _activities:
            if any(t in text for t in act["trigger"]):
                return _select_activity(act)
        # Try first-word match
        for act in _activities:
            if act["name"].lower().split()[0] in text:
                return _select_activity(act)
        return None

    if level == "category":
        for cat in _active["categories"]:
            if any(t in text for t in cat["trigger"]):
                if cat.get("topics"):
                    _active["selecting"] = "topic"
                    _active["topics"] = cat["topics"]
                    names = [t["name"] for t in cat["topics"]]
                    return f"I have {', '.join(names[:-1])}, or {names[-1]}. Which one?"
                elif cat.get("file"):
                    return _load_file(cat)
        return None

    if level == "topic":
        for topic in _active["topics"]:
            if any(t in text for t in topic["trigger"]):
                return _load_file(topic)
            if topic["name"].lower().split()[0] in text:
                return _load_file(topic)
        return None

    return None


def _select_activity(act) -> str:
    global _active
    # If it has categories, show them
    if "categories" in act:
        cats = act["categories"]
        _active = {"selecting": "category", "categories": cats}
        names = [c["name"] for c in cats]
        return f"I have {', '.join(names[:-1])}, or {names[-1]}. Which one?"
    # If it has a file directly, load it
    if "file" in act:
        return _load_file(act)
    # No special context needed (stories, jokes, etc.)
    _clear()
    return None


def _load_file(item) -> str:
    global _active
    filepath = os.path.join(_DIR, item["file"])
    with open(filepath) as f:
        data = yaml.safe_load(f)

    if "questions" in data:
        return _load_trivia(data)
    elif "items" in data:
        return _load_word_game(data)
    return None


def _load_trivia(data) -> str:
    global _active
    questions = data["questions"][:]
    random.shuffle(questions)

    qa_lines = []
    for q in questions:
        qa_lines.append(f"Q: {q['q']}")
        qa_lines.append(f"A: {q['a']}")
        qa_lines.append(f"Hint: {q['hint']}")
        qa_lines.append("")

    context = f"""ACTIVITY MODE: Trivia — {data['topic']}
{data['instructions']}

Use ONLY these questions and answers. Do NOT make up your own.
Ask them in the order listed below.

{chr(10).join(qa_lines)}

When the user says they're done, say something encouraging and end the quiz."""

    _active = {"context": context}
    return f"Great choice! Let's do {data['topic']} trivia."


def _load_word_game(data) -> str:
    global _active
    items = data["items"][:]
    random.shuffle(items)

    item_lines = []
    for item in items:
        item_lines.append(f"Answer: {item['answer']}")
        for i, clue in enumerate(item['clues'], 1):
            item_lines.append(f"  Clue {i}: {clue}")
        item_lines.append("")

    context = f"""ACTIVITY MODE: Word Game — {data['topic']}
{data['instructions']}

Use ONLY these items. Start with the first one. Give clues ONE at a time.

{chr(10).join(item_lines)}

When the user says they're done, say something encouraging and end the game."""

    _active = {"context": context}
    return f"Great! Let's play {data['topic']}."


def is_selecting() -> bool:
    return _active is not None and _active.get("selecting") is not None


def is_active() -> bool:
    return _active is not None and not _active.get("selecting")


def _clear():
    global _active
    _active = None
