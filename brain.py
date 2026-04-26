"""LLM interaction via Ollama/Together AI with streaming and persistent memory."""

import json
import os
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

# Together AI (cloud)
_TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not _TOGETHER_KEY:
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        for line in open(_env_path):
            if line.startswith("TOGETHER_API_KEY="):
                _TOGETHER_KEY = line.split("=", 1)[1].strip()

_TOGETHER_MODEL = _cfg.get("together", {}).get("model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
_TOGETHER_URL = "https://api.together.ai/v1/chat/completions"
_USE_TOGETHER = _cfg.get("together", {}).get("enabled", False) and bool(_TOGETHER_KEY)
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

# Church context — loaded once, injected when conversation touches on church topics
_church_context = None
try:
    with open("church_context.yaml") as _cf:
        _church_context = yaml.safe_load(_cf)
except FileNotFoundError:
    pass

_CHURCH_KEYWORDS = {"church", "conference", "general conference", "prophet", "elder",
                    "president oaks", "president nelson", "temple", "hymn", "hymns",
                    "scripture", "gospel", "faith", "prayer", "blessing", "sacrament",
                    "sunday", "bishop", "ward", "stake", "missionary", "apostle",
                    "latter-day", "lds", "heavenly father", "jesus", "christ",
                    "resurrection", "easter", "priesthood", "relief society"}


def _church_context_text() -> str:
    """Build a text summary of church context for the LLM."""
    if not _church_context:
        return ""
    lines = []
    cl = _church_context.get("current_leadership", {})
    if cl:
        lines.append(f"Current Church President: {cl.get('prophet', 'unknown')}.")
        lines.append(f"First Counselor: {cl.get('first_counselor')}, Second Counselor: {cl.get('second_counselor')}.")
        if cl.get("notes"):
            lines.append(cl["notes"].strip())
    conf = _church_context.get("april_2026_conference", {})
    if conf:
        lines.append(f"\nApril 2026 General Conference ({conf.get('date', '')}):")
        for h in conf.get("highlights", []):
            lines.append(f"- {h}")
        lines.append("\nKey talks:")
        for talk in conf.get("key_talks", []):
            lines.append(f"- {talk['speaker']}: \"{talk['title']}\" — {talk['summary'].strip()}")
        themes = conf.get("themes", [])
        if themes:
            lines.append(f"\nMain themes: {'; '.join(themes)}.")
    return "\n".join(lines)


# Speaker lookup for on-demand talk loading
_speakers = {}
try:
    with open("church/speakers.yaml") as _sf:
        _speakers = yaml.safe_load(_sf).get("speakers", {})
except FileNotFoundError:
    pass

# Church News headlines
_church_news = []
try:
    with open("church/news.json") as _nf:
        _news_data = json.load(_nf)
        _church_news = _news_data.get("headlines", [])
except (FileNotFoundError, json.JSONDecodeError):
    pass

_NEWS_KEYWORDS = {"news", "church news", "what's new", "what's happening", "latest",
                  "anything new", "what's going on"}


def _church_news_text() -> str:
    """Build a text summary of recent Church News headlines."""
    if not _church_news:
        return ""
    lines = ["Recent Church News headlines:"]
    for h in _church_news:
        title = h.get("title", "")
        if title and len(title) > 15:  # skip category-only entries
            lines.append(f"- {title}")
    return "\n".join(lines)


def _find_talk_for_message(msg: str) -> str | None:
    """If the user mentions a specific speaker, load their talk text."""
    msg_lower = msg.lower()
    for key, info in _speakers.items():
        if key in msg_lower or info["name"].split()[-1].lower() in msg_lower:
            talk_path = os.path.join("church", "talks", info["file"])
            if os.path.exists(talk_path):
                with open(talk_path, encoding="utf-8") as f:
                    text = f.read()
                # Truncate to ~4000 chars to fit context window
                if len(text) > 4000:
                    text = text[:4000] + "..."
                return f'{info["name"]} gave a talk called "{info["title"]}".\n\nHere is the content:\n{text}'
    return None


def _fetch_news_article(msg: str) -> str | None:
    """If the user's message matches a recent headline, fetch the full article."""
    import urllib.request, html as html_mod
    msg_lower = msg.lower()
    # Find best matching headline — check if key words from headline appear in message
    best = None
    best_score = 0
    for h in _church_news:
        title = h.get("title", "")
        url = h.get("url", "")
        if not title or not url or len(title) < 20:
            continue
        words = [w.lower() for w in re.split(r'\W+', title) if len(w) > 3]
        score = sum(1 for w in words if w in msg_lower)
        if score > best_score and score >= 2:
            best_score = score
            best = h
    if not best:
        return None
    try:
        req = urllib.request.Request(best["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Extract article text
        match = re.search(r'<article[^>]*>(.*?)</article>', raw, re.DOTALL)
        text = match.group(1) if match else raw
        # Strip HTML
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html_mod.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 4000:
            text = text[:4000] + "..."
        if len(text) > 200:
            return f'Article: "{best["title"]}"\n\n{text}'
    except Exception as e:
        print(f"[NEWS] Failed to fetch article: {e}")
    return None


def _build_system_prompt() -> str:
    now = datetime.now()
    time_ctx = f"Current date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}."
    parts = [_SYSTEM_PROMPT]

    # User profile
    if _user_profile:
        p = _user_profile
        profile_lines = [f"The person you are talking to is {p['name']} ({p['pronouns']})."]
        if p.get('age'):
            profile_lines.append(f"{p.get('nickname', p['name'])} is {p['age']} years old.")
        if p.get('location'):
            profile_lines.append(f"Lives in {p['location']}.")
        if p.get('hobbies'):
            profile_lines.append(f"Hobbies: {', '.join(p['hobbies'])}.")
        if p.get('health_notes'):
            profile_lines.append(f"Health: {p['health_notes']}")
        if p.get('personality'):
            profile_lines.append(f"Personality: {p['personality']}")
        favs = p.get('favorites', {})
        fav_parts = [f"{k}: {v}" for k, v in favs.items() if v]
        if fav_parts:
            profile_lines.append(f"Favorites: {', '.join(fav_parts)}.")
        # Family members
        for fm in p.get('family', []):
            line = f"{fm['name']} is her {fm['relationship']}"
            if fm.get('spouse'):
                line += f", married to {fm['spouse']}"
            if fm.get('location'):
                line += f", lives in {fm['location']}"
            if fm.get('notes'):
                line += f" ({fm['notes']})"
            profile_lines.append(line + ".")
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

    # Activity context
    try:
        from activities import get_activity_context
        act = get_activity_context()
        if act:
            parts.append(act)
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


def _stream_together(messages):
    """Stream from Together AI (OpenAI-compatible API)."""
    resp = requests.post(_TOGETHER_URL, json={
        "model": _TOGETHER_MODEL,
        "stream": True,
        "messages": messages,
    }, headers={"Authorization": f"Bearer {_TOGETHER_KEY}"}, stream=True, timeout=120)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        token = (data.get("choices") or [{}])[0].get("delta", {}).get("content", "")
        if token:
            yield token


def _stream_ollama(messages, url, model):
    """Stream from Ollama API."""
    resp = requests.post(url, json={
        "model": model,
        "stream": True,
        "messages": messages,
    }, stream=True, timeout=120)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("done"):
            break
        token = data.get("message", {}).get("content", "")
        if token:
            yield token


def ask_streaming(user_message: str, initiated_by: str = "user"):
    """Stream LLM response, yielding complete sentences as they arrive."""
    if initiated_by == "system":
        _history.append({"role": "user", "content": f"[You decided to say this to the user: {user_message}]"})
    else:
        _history.append({"role": "user", "content": user_message})

    trimmed = _history[-20:]
    messages = [{"role": "system", "content": _build_system_prompt()}] + trimmed

    # Inject church context if the user mentions church-related topics
    if _church_context:
        msg_lower = user_message.lower()
        if any(kw in msg_lower for kw in _CHURCH_KEYWORDS):
            # Check for specific speaker talk first
            talk_text = _find_talk_for_message(user_message)
            if talk_text:
                messages.insert(1, {"role": "system", "content":
                    f"The user is asking about a specific conference talk. Here it is:\n{talk_text}\n"
                    "Discuss this talk naturally. Share key points and stories from it. "
                    "Don't read it word for word — summarize and discuss like a friend would."})
            # Check for church news request
            elif _church_news:
                # First check if user is asking about a specific headline
                article = _fetch_news_article(user_message)
                if article:
                    messages.insert(1, {"role": "system", "content":
                        f"The user is asking about a church news article. Here is the full article:\n{article}\n"
                        "Discuss this article naturally. Share key points like a friend would. "
                        "Don't read it word for word — summarize the main points."})
                elif any(kw in msg_lower for kw in _NEWS_KEYWORDS):
                    news_text = _church_news_text()
                    messages.insert(1, {"role": "system", "content":
                        f"The user is asking about church news. Here are recent headlines:\n{news_text}\n"
                        "Share a few interesting headlines naturally. Don't list them all — "
                        "pick 2-3 that would interest an older Latter-day Saint woman.\n"
                        "IMPORTANT: You ONLY know the headlines, not the article details. "
                        "If the user asks for more detail about a headline, say you only saw "
                        "the headline and suggest they ask a family member or check the Church News "
                        "website for the full story. NEVER make up details about a news story."})
            else:
                ctx = _church_context_text()
                if ctx:
                    messages.insert(1, {"role": "system", "content":
                        f"The user wants to discuss church topics. Here is current context:\n{ctx}\n"
                        "Use this information naturally in conversation. Don't recite it all at once — "
                        "share relevant bits as the conversation flows."})

    # Try providers in order: Together AI → Ollama primary → Ollama fallback
    token_stream = None
    for attempt in ["together", "ollama", "fallback"]:
        try:
            if attempt == "together" and _USE_TOGETHER:
                token_stream = _stream_together(messages)
            elif attempt == "ollama":
                token_stream = _stream_ollama(messages, _OLLAMA_URL, _MODEL)
            elif attempt == "fallback" and _OLLAMA_FALLBACK:
                print(f"[LLM] Falling back to {_OLLAMA_FALLBACK}")
                token_stream = _stream_ollama(messages, _OLLAMA_FALLBACK + "/api/chat", _FALLBACK_MODEL)
            else:
                continue
            # Test the stream by getting first token
            break
        except Exception as e:
            print(f"[LLM] {attempt} failed: {e}")
            token_stream = None

    if token_stream is None:
        yield "I'm having trouble thinking right now. Can you try again?"
        return

    buffer = ""
    full_reply = ""

    try:
      for token in token_stream:
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
    except Exception as e:
      print(f"[LLM] Stream error: {e}")

    leftover = buffer.strip()
    if leftover:
        full_reply += leftover
        yield leftover

    _history.append({"role": "assistant", "content": full_reply.strip()})

    # Extract memories in background
    import threading
    threading.Thread(target=_extract_memories, args=(user_message, full_reply.strip()), daemon=True).start()
