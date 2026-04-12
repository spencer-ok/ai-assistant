"""Entry point: reactive voice loop + proactive triggers + web UI."""

import sys
import threading
import time as _time

import yaml

from brain import ask_streaming
import voice
import proactive
from ui.app import app as flask_app, set_status, add_transcript, get_text_input, should_stop, _stop_flag, _text_input_queue, is_mic_muted, get_caregiver_context, get_unread_notes

# Log to file
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                s.write(data.encode('ascii', errors='replace').decode('ascii'))
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_logfile = open("rosie.log", "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _logfile)
sys.stderr = _Tee(sys.stderr, _logfile)

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_NAME = _cfg["persona"]["name"]
_DUPLEX = _cfg["voice"].get("duplex", False)
_EXIT_PHRASES = {"terminate rosie application"}
_MAX_SENTENCES = 0


def _ts():
    return _time.strftime("%H:%M:%S", _time.localtime()) + f".{int(_time.time()*1000)%1000:03d}"


def _is_exit(text: str) -> bool:
    t = text.strip().lower()
    return "terminate" in t and ("rosie" in t or "rosy" in t or "rosi" in t) and "application" in t


def _speak_streaming(message: str, initiated_by: str = "user"):
    # Deliver any unread caregiver notes directly via TTS (don't rely on LLM)
    # Start LLM request in parallel if there are notes
    notes = get_unread_notes()
    if notes:
        from ui.app import _caregiver_profiles
        import threading

        # Start LLM streaming in background while we speak the note
        llm_sentences = []
        llm_done = threading.Event()
        def _prefetch_llm():
            for sentence in ask_streaming(message, initiated_by=initiated_by):
                llm_sentences.append(sentence)
            llm_done.set()
        threading.Thread(target=_prefetch_llm, daemon=True).start()

        # Greet first, then deliver notes
        from brain import _user_profile
        user_name = _user_profile.get("name", "") if _user_profile else ""
        greeting = f"Hello {user_name}!" if user_name else "Hello!"
        set_status("speaking", greeting)
        add_transcript("rosie", greeting)
        voice.speak(greeting, stop_check=lambda: _stop_flag.is_set())

        for n in notes:
            profile = next((p for p in _caregiver_profiles if p["name"].lower() == n["from"].lower()), None)
            if profile:
                pronoun = "He" if "he" in profile["pronouns"] else "She"
                note_msg = f'{n["from"]} left you a message. {pronoun} said: {n["text"]}'
            else:
                note_msg = f'{n["from"]} left you a message: {n["text"]}'
            print(f"[{_ts()}][NOTE] Speaking: {note_msg}")
            set_status("speaking", note_msg)
            add_transcript("rosie", note_msg)
            voice.speak(note_msg, stop_check=lambda: _stop_flag.is_set())

        # Now speak the pre-fetched LLM response
        print(f"[{_ts()}][LLM] Using prefetched response...")
        first = True
        count = 0
        full_reply = ""
        # Speak sentences that already arrived
        for sentence in llm_sentences:
            if first:
                print(f"[{_ts()}][LLM] First sentence received")
                first = False
            full_reply += sentence + " "
            set_status("speaking", sentence)
            add_transcript("rosie", full_reply.strip(), update_last=(count > 0))
            voice.speak(sentence, stop_check=lambda: _stop_flag.is_set())
            if _stop_flag.is_set():
                _stop_flag.clear()
                add_transcript("user", "[Stopped]")
                break
            count += 1
        # Wait for any remaining sentences
        if not _stop_flag.is_set():
            llm_done.wait(timeout=30)
            for sentence in llm_sentences[count:]:
                full_reply += sentence + " "
                set_status("speaking", sentence)
                add_transcript("rosie", full_reply.strip(), update_last=True)
                voice.speak(sentence, stop_check=lambda: _stop_flag.is_set())
                if _stop_flag.is_set():
                    _stop_flag.clear()
                    add_transcript("user", "[Stopped]")
                    break
        set_status("idle")
        print(f"[{_ts()}][LLM] Done")
        return

    print(f"[{_ts()}][LLM] Requesting response...")
    set_status("thinking")
    first = True
    count = 0
    full_reply = ""
    add_transcript("rosie", "...")  # placeholder that gets updated
    for sentence in ask_streaming(message, initiated_by=initiated_by):
        if first:
            print(f"[{_ts()}][LLM] First sentence received")
            first = False
        if should_stop():
            print(f"[{_ts()}][STOP] User pressed stop")
            add_transcript("user", "[Stopped]")
            break
        full_reply += sentence + " "
        set_status("speaking", sentence)
        add_transcript("rosie", full_reply.strip(), update_last=True)
        voice.speak(sentence, stop_check=lambda: _stop_flag.is_set())
        if _stop_flag.is_set():
            _stop_flag.clear()
            print(f"[{_ts()}][STOP] User pressed stop")
            add_transcript("user", "[Stopped]")
            break
        count += 1
        if _MAX_SENTENCES and count >= _MAX_SENTENCES:
            print(f"[{_ts()}][LLM] Sentence limit reached, stopping")
            break
    set_status("idle")
    print(f"[{_ts()}][LLM] Done")


def _deliver_unread_notes():
    """After user initiates conversation, deliver any pending caregiver notes."""
    notes = get_unread_notes()
    if not notes:
        return
    from ui.app import _caregiver_profiles
    parts = []
    for n in notes:
        # Find profile for context
        profile = next((p for p in _caregiver_profiles if p["name"].lower() == n["from"].lower()), None)
        if profile:
            parts.append(f'{n["from"]} (the user\'s {profile["relationship"]}, use {profile["pronouns"]}) says: "{n["text"]}"')
        else:
            parts.append(f'{n["from"]} says: "{n["text"]}"')
    summary = "; ".join(parts)
    msg = f'[System: You just greeted the user. Now share these messages naturally without re-greeting: {summary}]'
    print(f"[{_ts()}][NOTE] Delivering {len(notes)} unread note(s)")
    _speak_streaming(msg, initiated_by="system")


def _shutdown():
    set_status("speaking", "Goodbye! Talk to you later.")
    voice.speak("Goodbye! Talk to you later.")
    print(f"[EXIT] {_NAME} shutting down.")
    import os
    os._exit(0)


def handle_proactive(message: str):
    _speak_streaming(message, initiated_by="system")


def reactive_loop():
    while True:
        # Check for text input from UI
        text_input = get_text_input()
        if text_input:
            is_system = text_input.startswith("[System:")
            if not is_system:
                print(f"[{_ts()}][TXT INPUT] {text_input}")
                add_transcript("user", text_input)
            else:
                print(f"[{_ts()}][NOTE] {text_input}")
            if _is_exit(text_input):
                _shutdown()
            # Check activity triggers for text input too
            from activities import check_trigger, check_trivia_selection, is_selecting, is_active
            if is_selecting():
                resp = check_trivia_selection(text_input)
                if resp:
                    print(f"[{_ts()}][ACTIVITY] {resp}")
                    set_status("speaking", resp)
                    add_transcript("rosie", resp)
                    voice.speak(resp, stop_check=lambda: _stop_flag.is_set())
                    if is_active():
                        _speak_streaming("Ask me the first trivia question.", initiated_by="user")
                    set_status("idle")
                    continue
            trigger_resp = check_trigger(text_input)
            if trigger_resp:
                print(f"[{_ts()}][ACTIVITY] {trigger_resp}")
                set_status("speaking", trigger_resp)
                add_transcript("rosie", trigger_resp)
                voice.speak(trigger_resp, stop_check=lambda: _stop_flag.is_set())
                set_status("idle")
                continue
            _speak_streaming(text_input, initiated_by="system" if is_system else "user")
            continue

        # Skip mic if muted or text is queued
        if is_mic_muted() or len(_text_input_queue) > 0:
            import time as t
            t.sleep(0.3)
            continue

        # Voice input
        set_status("listening")
        user_input = voice.listen(text_pending_check=lambda: len(_text_input_queue) > 0)
        if user_input:
            print(f"[{_ts()}][VOICE INPUT] {user_input}")
            add_transcript("user", user_input)
            if _is_exit(user_input):
                _shutdown()
            # Check activity triggers
            from activities import check_trigger, check_trivia_selection, is_selecting, is_active
            if is_selecting():
                resp = check_trivia_selection(user_input)
                if resp:
                    print(f"[{_ts()}][ACTIVITY] {resp}")
                    set_status("speaking", resp)
                    add_transcript("rosie", resp)
                    voice.speak(resp, stop_check=lambda: _stop_flag.is_set())
                    # If trivia just loaded, ask first question immediately
                    if is_active():
                        _speak_streaming("Ask me the first trivia question.", initiated_by="user")
                    set_status("idle")
                    continue
            trigger_resp = check_trigger(user_input)
            if trigger_resp:
                print(f"[{_ts()}][ACTIVITY] {trigger_resp}")
                set_status("speaking", trigger_resp)
                add_transcript("rosie", trigger_resp)
                voice.speak(trigger_resp, stop_check=lambda: _stop_flag.is_set())
                set_status("idle")
                continue
            _speak_streaming(user_input, initiated_by="user")
        else:
            set_status("idle")


if __name__ == "__main__":
    threading.Thread(target=proactive.start, args=(handle_proactive,), daemon=True).start()
    threading.Thread(target=flask_app.run, kwargs={"host": "0.0.0.0", "port": 5000, "debug": False, "use_reloader": False}, daemon=True).start()
    threading.Thread(target=voice.debug_log_loop, daemon=True).start()
    print(f"[READY] {_NAME} is ready. UI at http://localhost:5000")
    reactive_loop()
