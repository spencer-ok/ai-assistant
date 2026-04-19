"""Flask UI for Rosie — status display, transcript, text input, stop button."""

import threading
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Shared state — set by main.py
_state = {
    "status": "idle",       # idle, listening, thinking, speaking
    "transcript": [],       # list of {"role": "user"|"rosie", "text": "..."}
    "current_speech": "",   # what Rosie is currently saying
}
_state_lock = threading.Lock()
_text_input_queue = []
_stop_flag = threading.Event()
_mic_muted = threading.Event()

# Caregiver messages
_caregiver_messages: list[dict] = []
_unread_notes: list[dict] = []
_caregiver_lock = threading.Lock()

# Load caregiver profiles
import yaml as _yaml, os as _os
_CAREGIVERS_FILE = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "caregivers.yaml")
_caregiver_profiles = []
if _os.path.exists(_CAREGIVERS_FILE):
    with open(_CAREGIVERS_FILE) as _f:
        _caregiver_profiles = _yaml.safe_load(_f).get("caregivers", [])


def set_status(status, speech=""):
    with _state_lock:
        _state["status"] = status
        _state["current_speech"] = speech


def add_transcript(role, text, update_last=False):
    with _state_lock:
        if update_last and _state["transcript"] and _state["transcript"][-1]["role"] == role:
            _state["transcript"][-1]["text"] = text
        else:
            _state["transcript"].append({"role": role, "text": text})
            if len(_state["transcript"]) > 50:
                _state["transcript"] = _state["transcript"][-50:]


def get_text_input():
    """Return queued text input, or None. Clears all queued to avoid duplicates."""
    if _text_input_queue:
        text = _text_input_queue.pop(0)
        _text_input_queue.clear()  # discard any duplicates
        return text
    return None


def should_stop():
    if _stop_flag.is_set():
        _stop_flag.clear()
        return True
    return False


def is_mic_muted():
    import voice
    return voice.get_listen_mode() == "muted"


def get_unread_notes() -> list[dict]:
    """Pop unread notes for delivery after user initiates conversation."""
    with _caregiver_lock:
        notes = list(_unread_notes)
        _unread_notes.clear()
    return notes


def get_caregiver_context() -> str:
    """Return caregiver profiles and recent messages for the LLM."""
    parts = []

    # Profiles
    if _caregiver_profiles:
        profiles = []
        for p in _caregiver_profiles:
            profiles.append(f"- {p['name']}: {p['relationship']} ({p['pronouns']}). {p.get('notes', '')}")
        parts.append("Family/caregivers you know about:\n" + "\n".join(profiles))

    # Recent messages
    with _caregiver_lock:
        if _caregiver_messages:
            from datetime import datetime
            now = datetime.now()
            lines = []
            for m in _caregiver_messages[-10:]:
                line = f"- {m['from']} said: \"{m['text']}\" (note left at {m['time']})"
                # Calculate how long ago
                try:
                    note_time = datetime.strptime(m['time'], "%I:%M %p").replace(
                        year=now.year, month=now.month, day=now.day)
                    diff = now - note_time
                    mins = int(diff.total_seconds() / 60)
                    if 0 < mins < 600:
                        line += f" — that was about {mins} minutes ago"
                except Exception:
                    pass
                lines.append(line)
            parts.append("Recent notes from family/caregivers (use naturally, don't announce unless asked or relevant):\n" + "\n".join(lines))

    return "\n\n".join(parts)


@app.route("/")
def index():
    theme = request.args.get("theme", "robot")
    return render_template(f"themes/{theme}.html")


@app.route("/themes")
def themes():
    return jsonify({"themes": ["default", "robot"]})


@app.route("/state")
def state():
    import voice
    with _state_lock:
        s = dict(_state)
        s["listen_mode"] = voice.get_listen_mode()
        s["mic_muted"] = voice.get_listen_mode() == "muted"
        return jsonify(s)


@app.route("/send", methods=["POST"])
def send():
    text = request.json.get("text", "").strip()
    if text:
        _text_input_queue.append(text)
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    _stop_flag.set()
    return jsonify({"ok": True})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    _text_input_queue.append("Terminate Rosie Application")
    return jsonify({"ok": True})


@app.route("/mute", methods=["POST"])
def mute():
    import voice
    mode = request.json.get("mode") if request.is_json else None
    if mode:
        voice.set_listen_mode(mode)
    else:
        # Cycle: always -> name -> muted -> always
        cur = voice.get_listen_mode()
        nxt = {"always": "name", "name": "muted", "muted": "always"}[cur]
        voice.set_listen_mode(nxt)
    return jsonify({"listen_mode": voice.get_listen_mode()})


@app.route("/caregivers")
def caregivers_list():
    return jsonify({"caregivers": _caregiver_profiles})


@app.route("/caregiver", methods=["GET", "POST"])
def caregiver():
    if request.method == "POST":
        text = request.json.get("text", "").strip()
        sender = request.json.get("from", "Caregiver").strip()
        speak = request.json.get("speak", True)
        if text:
            from datetime import datetime
            with _caregiver_lock:
                _caregiver_messages.append({
                    "from": sender,
                    "text": text,
                    "time": datetime.now().strftime("%I:%M %p"),
                })
                if len(_caregiver_messages) > 20:
                    _caregiver_messages[:] = _caregiver_messages[-20:]
            if speak:
                with _caregiver_lock:
                    _unread_notes.append({"from": sender, "text": text})
        return jsonify({"ok": True})
    else:
        with _caregiver_lock:
            return jsonify({"messages": list(_caregiver_messages)})


@app.route("/caregiver/record", methods=["POST"])
def caregiver_record():
    """Record a note by redirecting the main mic stream."""
    sender = request.json.get("from", "Caregiver") if request.is_json else "Caregiver"

    import time as _t, numpy as np, whisper as _w
    import voice

    # Start capturing from main mic
    with voice._cg_lock:
        voice._cg_buffer.clear()
    voice._cg_recording.set()
    _t.sleep(8)
    voice._cg_recording.clear()

    with voice._cg_lock:
        chunks = list(voice._cg_buffer)
        voice._cg_buffer.clear()

    if not chunks:
        return jsonify({"text": ""})

    audio_f32 = np.concatenate(chunks).astype(np.float32) / 32768.0
    if np.sqrt(np.mean(audio_f32 ** 2)) < 0.001:
        return jsonify({"text": ""})

    _rec_model = _w.load_model("tiny")
    result = _rec_model.transcribe(audio_f32, fp16=False, language="en")
    text = result["text"].strip()

    if text:
        from datetime import datetime
        with _caregiver_lock:
            _caregiver_messages.append({
                "from": sender,
                "text": text,
                "time": datetime.now().strftime("%I:%M %p"),
            })
            _unread_notes.append({"from": sender, "text": text})
    return jsonify({"text": text or ""})


def start(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)
