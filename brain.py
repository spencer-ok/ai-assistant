"""LLM interaction via Ollama with streaming and persistent memory."""

import json
import re
from datetime import datetime

import requests
import yaml

import memory

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_OLLAMA_HOST = _cfg["ollama"]["host"]
_OLLAMA_FALLBACK = _cfg["ollama"].get("fallback_host", None)
_OLLAMA_URL = _OLLAMA_HOST + "/api/chat"
_MODEL = _cfg["persona"]["model"]
_FALLBACK_MODEL = _cfg["ollama"].get("fallback_model", _MODEL)
_SYSTEM_PROMPT = _cfg["persona"]["system_prompt"]
_history: list[dict] = []

_SENTENCE_END = re.compile(r'[.!?]\s')

_EXTRACT_PROMPT = (
    "You extract facts from conversations. "
    "List what you learned about the user as a JSON array of strings. "
    "Always include people mentioned, activities, preferences, health, and plans. "
    'Return ONLY valid JSON. Example: ["went grocery shopping", "has a daughter named Sarah"]\n\n'
)


_user_profile = None
try:
    with open("profile.yaml") as _pf:
        _user_profile = yaml.safe_load(_pf).get("user", {})
except FileNotFoundError:
    pass


def _build_system_prompt() -> str:
    now = datetime.now()
    time_ctx = f"Current date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}."
    parts = [_SYSTEM_PROMPT]

    # User profile
    if _user_profile:
        p = _user_profile
        profile_lines = [f"The person you are talking to is {p['name']} ({p['pronouns']})."]
        if p.get('age'):
            profile_lines.append(f"{p['name']} is {p['age']} years old.")
        if p.get('hobbies'):
            profile_lines.append(f"Hobbies: {', '.join(p['hobbies'])}.")
        if p.get('health_notes'):
            profile_lines.append(f"Health: {p['health_notes']}")
        favs = p.get('favorites', {})
        fav_parts = [f"{k}: {v}" for k, v in favs.items() if v]
        if fav_parts:
            profile_lines.append(f"Favorites: {', '.join(fav_parts)}.")
        if p.get('notes'):
            profile_lines.append(p['notes'])
        profile_lines.append(f"Always call the user {p.get('nickname', p['name'])}. Never call them Rosie.")
        parts.append("\n".join(profile_lines))

    parts.append(time_ctx)

    mem = memory.get_recent()
    if mem:
        parts.append(mem)
    # Caregiver context
    try:
        from ui.app import get_caregiver_context
        cg = get_caregiver_context()
        if cg:
            parts.append(cg)
    except ImportError:
        pass
    return "\n\n".join(parts)


def _extract_memories(user_msg: str, assistant_msg: str):
    """Ask the LLM to extract notable facts, then store them."""
    try:
        resp = requests.post(_OLLAMA_URL, json={
            "model": _MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user", "content": f"User said: {user_msg}\nAssistant said: {assistant_msg}"},
            ],
        })
        text = resp.json()["message"]["content"].strip()
        # Find JSON array in response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            for fact in facts:
                if isinstance(fact, str) and len(fact) > 5:
                    memory.add(fact)
    except Exception:
        pass  # Memory extraction is best-effort


def ask(user_message: str, initiated_by: str = "user") -> str:
    """Non-streaming fallback."""
    return "".join(chunk for chunk in ask_streaming(user_message, initiated_by))


def ask_streaming(user_message: str, initiated_by: str = "user"):
    """Stream LLM response, yielding complete sentences as they arrive."""
    if initiated_by == "system":
        _history.append({"role": "user", "content": f"[You decided to say this to the user: {user_message}]"})
    else:
        _history.append({"role": "user", "content": user_message})

    trimmed = _history[-20:]

    url = _OLLAMA_URL
    model = _MODEL
    try:
        resp = requests.post(url, json={
            "model": model,
            "stream": True,
            "messages": [{"role": "system", "content": _build_system_prompt()}] + trimmed,
        }, stream=True, timeout=120)
        resp.raise_for_status()
    except Exception:
        if _OLLAMA_FALLBACK:
            print(f"[LLM] Primary failed, falling back to {_OLLAMA_FALLBACK}")
            url = _OLLAMA_FALLBACK + "/api/chat"
            model = _FALLBACK_MODEL
            resp = requests.post(url, json={
                "model": model,
                "stream": True,
                "messages": [{"role": "system", "content": _build_system_prompt()}] + trimmed,
            }, stream=True, timeout=120)
            resp.raise_for_status()
        else:
            raise

    buffer = ""
    full_reply = ""

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("done"):
            break
        token = data.get("message", {}).get("content", "")
        buffer += token

        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break
            idx = match.end()
            sentence = buffer[:idx].strip()
            buffer = buffer[idx:]
            if sentence:
                full_reply += sentence + " "
                yield sentence

    leftover = buffer.strip()
    if leftover:
        full_reply += leftover
        yield leftover

    _history.append({"role": "assistant", "content": full_reply.strip()})

    # Extract memories in background
    import threading
    threading.Thread(target=_extract_memories, args=(user_message, full_reply.strip()), daemon=True).start()
