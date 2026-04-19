"""Speech-to-Text (Whisper) and Text-to-Speech (Piper) with always-on mic and AEC."""

import collections
import threading
import time

import numpy as np
import sounddevice as sd
import whisper
import yaml
from piper.voice import PiperVoice
from aec import EchoCanceller

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)["voice"]

_whisper_model = whisper.load_model(_cfg["whisper_model"])
_whisper_lock = threading.Lock()
_piper_voice = PiperVoice.load(_cfg["piper_model"])
_SILENCE_DUR = _cfg["silence_duration"]
_DUPLEX = _cfg.get("duplex", False)
_SAMPLE_RATE = 16000
_CHUNK_SEC = 0.5
_CHUNK_SAMPLES = int(_SAMPLE_RATE * _CHUNK_SEC)
_INPUT_DEV = _cfg.get("input_device", None)
_OUTPUT_DEV = _cfg.get("output_device", None)
_DEBUG_MIC = _cfg.get("debug_mic", None)  # secondary mic for ground truth recording

# Listen modes: "always", "name", "muted"
_listen_mode = _cfg.get("listen_mode", "always")
_listen_lock = threading.Lock()

# Wake word conversation window
_WAKE_TIMEOUT = 30  # seconds of silence before returning to wake word mode
_wake_active_until = 0  # timestamp when conversation window expires

# OpenWakeWord for name-activated mode
from openwakeword.model import Model as _OWWModel
_oww_model = _OWWModel(wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx")
_oww_threshold = 0.5
_wake_detected = threading.Event()

# AEC — longer filter for TV/HDMI delay (16k * 0.3s = 4800 samples across blocks)
_ECHO_DELAY = _cfg.get("echo_delay", 0)
_aec = EchoCanceller(block_size=512, filter_blocks=8, mu=0.5) if _DUPLEX else None
_playback_ring = collections.deque(maxlen=_CHUNK_SAMPLES * 8 + _ECHO_DELAY)  # enough history for delay
_playback_lock = threading.Lock()

# Always-on mic state
_mic_buffer = collections.deque(maxlen=120)
_mic_lock = threading.Lock()
_speech_event = threading.Event()
_chunk_counter = 0

# Caregiver recording mode
_cg_recording = threading.Event()
_cg_buffer = []
_cg_lock = threading.Lock()

# Playback state
_interrupted = threading.Event()
_is_speaking = threading.Event()
_playback_done = threading.Event()

_SILENCE_THRESH = 30


def _ts():
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time()*1000)%1000:03d}"


_dev_info = sd.query_devices(_INPUT_DEV)
_mic_channels = min(_dev_info.get("max_input_channels", 1), 4) or 1

def _mic_callback(indata, frames, time_info, status):
    global _chunk_counter
    chunk = indata[:, 0].copy()

    # Redirect to caregiver recording buffer if active
    if _cg_recording.is_set():
        with _cg_lock:
            _cg_buffer.append(chunk)
        return

    with _mic_lock:
        _mic_buffer.append((_chunk_counter, chunk))
        _chunk_counter += 1

    # In name-activated mode, feed OpenWakeWord instead of RMS detection
    if _listen_mode == "name" and time.time() > _wake_active_until:
        # OWW expects int16 numpy array
        prediction = _oww_model.predict(chunk)
        for mdl in _oww_model.prediction_buffer.keys():
            scores = list(_oww_model.prediction_buffer[mdl])
            if scores and scores[-1] > _oww_threshold:
                _wake_detected.set()
                _speech_event.set()
                _oww_model.reset()
                return
        return

    # In muted mode, just buffer (for caregiver recording) but don't detect
    if _listen_mode == "muted":
        return

    # Always-listening mode (or name mode after wake detected): RMS-based detection
    if _aec is not None and _is_speaking.is_set():
        with _playback_lock:
            ref_samples = list(_playback_ring)
        offset = len(ref_samples) - len(chunk) - _ECHO_DELAY
        if offset >= 0:
            ref = np.array(ref_samples[offset:offset + len(chunk)], dtype=np.int16)
            cleaned = _aec.process_chunk(chunk, ref)
            rms = np.sqrt(np.mean(cleaned ** 2)) * 32768.0
        else:
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
    else:
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

    if rms >= _SILENCE_THRESH:
        _speech_event.set()


_mic_stream = sd.InputStream(
    samplerate=_SAMPLE_RATE, channels=_mic_channels, dtype="int16",
    blocksize=_CHUNK_SAMPLES, callback=_mic_callback,
    device=_INPUT_DEV,
)
_mic_stream.start()

# Debug mic — records clean user input for comparison
_debug_buffer = collections.deque(maxlen=120)
_debug_lock = threading.Lock()
_debug_counter = 0

def _debug_mic_callback(indata, frames, time_info, status):
    global _debug_counter
    chunk = indata[:, 0].copy()
    with _debug_lock:
        _debug_buffer.append((_debug_counter, chunk))
        _debug_counter += 1

if _DEBUG_MIC is not None:
    _debug_stream = sd.InputStream(
        samplerate=_SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=_CHUNK_SAMPLES, callback=_debug_mic_callback,
        device=_DEBUG_MIC,
    )
    _debug_stream.start()
    print(f"[DBG] Debug mic active on device {_DEBUG_MIC}")


def calibrate():
    global _SILENCE_THRESH
    print(f"[{_ts()}][CAL] Calibrating mic - stay quiet...")
    time.sleep(2)
    with _mic_lock:
        recent = [chunk for _, chunk in list(_mic_buffer)[-4:]]
    if recent:
        levels = [np.sqrt(np.mean(c.astype(np.float32) ** 2)) for c in recent]
        ambient = max(np.median(levels), 10)
        _SILENCE_THRESH = min(ambient * 3, 500)
    print(f"[{_ts()}][CAL] Threshold: {_SILENCE_THRESH:.0f}")


calibrate()


def _drain_speech() -> np.ndarray | None:
    # Start from a couple chunks BEFORE the trigger to catch the beginning
    with _mic_lock:
        buf = list(_mic_buffer)
    
    # Find the chunk that triggered speech and include 2 before it
    start_idx = max(0, len(buf) - 3)
    chunks = [c for _, c in buf[start_idx:]]
    last_seen = buf[-1][0] if buf else _chunk_counter

    print(f"[{_ts()}][MIC] Speech detected, pre-buffered {len(chunks)} chunks")

    silent_chunks = 0
    max_silent = int(_SILENCE_DUR / _CHUNK_SEC)
    timeout = time.time() + 30

    while time.time() < timeout:
        time.sleep(0.05)
        with _mic_lock:
            new_chunks = [(i, c) for i, c in _mic_buffer if i > last_seen]

        if not new_chunks:
            continue

        for idx, chunk in new_chunks:
            last_seen = idx
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            if rms < _SILENCE_THRESH:
                silent_chunks += 1
                if silent_chunks >= max_silent:
                    print(f"[{_ts()}][MIC] Silence detected, {len(chunks)} chunks total")
                    return np.concatenate(chunks).astype(np.float32) / 32768.0
                # Keep silent chunks during speech (pauses between words)
                chunks.append(chunk)
            else:
                silent_chunks = 0
                chunks.append(chunk)

    if not chunks:
        return None
    print(f"[{_ts()}][MIC] Max time reached, {len(chunks)} chunks")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _transcribe(audio: np.ndarray) -> str | None:
    t0 = time.perf_counter()
    with _whisper_lock:
        result = _whisper_model.transcribe(
            audio, fp16=False, language="en",
            initial_prompt="This is a conversation with an older adult. Short replies like yes, no, okay, thank you, good morning.",
        )
    elapsed = time.perf_counter() - t0
    text = result["text"].strip()
    print(f"[{_ts()}][STT] ({elapsed:.2f}s) {text}")
    return text if text else None


_recalibrate_interval = 60  # seconds
_last_recalibrate = time.time()


def _maybe_recalibrate():
    """Periodically update threshold from recent quiet chunks."""
    global _SILENCE_THRESH, _last_recalibrate
    if time.time() - _last_recalibrate < _recalibrate_interval:
        return
    _last_recalibrate = time.time()
    with _mic_lock:
        recent = [c for _, c in list(_mic_buffer)[-8:]]
    if not recent:
        return
    levels = [np.sqrt(np.mean(c.astype(np.float32) ** 2)) for c in recent]
    # Only recalibrate if it looks quiet (no speech in recent chunks)
    median = np.median(levels)
    if median < _SILENCE_THRESH * 0.5:  # only adjust from truly quiet baseline
        new_thresh = min(max(median * 3, 30), 500)
        if abs(new_thresh - _SILENCE_THRESH) > 10:
            print(f"[{_ts()}][CAL] Recalibrated: {_SILENCE_THRESH:.0f} -> {new_thresh:.0f}")
            _SILENCE_THRESH = new_thresh


def listen(text_pending_check=None) -> str | None:
    """Wait for speech, record until silence, transcribe.
    Returns None immediately if text_pending_check() returns True."""
    global _wake_active_until
    if _listen_mode == "muted":
        import time as _t
        _t.sleep(0.3)
        return None

    # Flush stale audio from mic buffer to avoid echo bleed
    _speech_event.clear()
    _oww_model.reset()
    import time as _t
    _t.sleep(0.5)
    with _mic_lock:
        _mic_buffer.clear()
    _speech_event.clear()

    # In name mode: check if conversation window is still active
    in_conversation = _listen_mode == "name" and time.time() < _wake_active_until
    if _listen_mode == "name" and not in_conversation:
        _wake_detected.clear()
        print(f"[{_ts()}][MIC] Waiting for wake word...")
    else:
        print(f"[{_ts()}][MIC] Listening...")

    # Wait for speech, but bail out if text input arrives or mic is muted
    while not _speech_event.wait(timeout=0.3):
        if text_pending_check and text_pending_check():
            return None
        if _listen_mode == "always":
            _maybe_recalibrate()

    if _wake_detected.is_set() and not in_conversation:
        print(f"[{_ts()}][WAKE] Wake word detected!")
        _wake_active_until = time.time() + _WAKE_TIMEOUT

    # Check if we got muted while waiting
    from ui.app import is_mic_muted
    if is_mic_muted() or _listen_mode == "muted":
        _speech_event.clear()
        return None

    audio = _drain_speech()
    if audio is None:
        return None

    # Check again after recording — might have been muted mid-speech
    if is_mic_muted():
        return None
    text = _transcribe(audio)

    # Keep conversation window open after each utterance
    if text and _listen_mode == "name":
        _wake_active_until = time.time() + _WAKE_TIMEOUT

    # Also transcribe debug mic for comparison
    if _DEBUG_MIC is not None and text:
        with _debug_lock:
            debug_chunks = [c for _, c in list(_debug_buffer)[-20:]]
        if debug_chunks:
            debug_audio = np.concatenate(debug_chunks).astype(np.float32) / 32768.0
            debug_result = _whisper_model.transcribe(debug_audio, fp16=False, language="en",
                initial_prompt="This is a conversation with an older adult. Short replies like yes, no, okay, thank you, good morning.")
            debug_text = debug_result["text"].strip()
            print(f"[{_ts()}][DBG] Clean mic heard: {debug_text}")

    return text


def debug_log_loop():
    """Periodically transcribe debug mic to show what it hears, even during TTS."""
    if _DEBUG_MIC is None:
        return
    _debug_whisper = whisper.load_model("tiny")  # separate model for thread safety
    while True:
        time.sleep(2)
        with _debug_lock:
            chunks = [c for _, c in list(_debug_buffer)[-4:]]
        if not chunks:
            continue
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio ** 2)) * 32768.0
        if rms < 50:
            continue
        try:
            result = _debug_whisper.transcribe(audio, fp16=False, language="en",
                initial_prompt="This is a conversation with an older adult.")
            text = result["text"].strip()
            if text:
                print(f"[{_ts()}][DBG] ({rms:.0f}) {text}")
        except Exception:
            pass


import re as _re

def _tts_cleanup(text: str) -> str:
    """Fix common patterns that TTS reads wrong."""
    # Convert 4-digit times like 1130 to "11 30"
    def _fix_time(m):
        h = int(m.group(1))
        mins = m.group(2)
        # Use current hour to guess AM/PM
        from datetime import datetime
        now_h = datetime.now().hour
        if h <= 12:
            # If it's currently PM and the time is <= 12, assume PM
            if now_h >= 12:
                ampm = " PM"
            else:
                ampm = " AM"
        else:
            h -= 12
            ampm = " PM"
        return f"{h}:{mins}{ampm}" if mins != "00" else f"{h}{ampm}"
    text = _re.sub(r'(?<![,\d])([12]?\d)((?:00|15|30|45))(?![,\d])', lambda m: _fix_time(m) if 1 <= int(m.group(1)) <= 24 else m.group(0), text)
    # Also handle "11.30" format
    text = _re.sub(r'(?<![,\d])([12]?\d)\.(00|15|30|45)(?![,\d])', lambda m: _fix_time(m) if 1 <= int(m.group(1)) <= 24 else m.group(0), text)
    return text


def speak(text: str, stop_check=None) -> None:
    text = _tts_cleanup(text)
    t0 = time.perf_counter()
    chunks = list(_piper_voice.synthesize(text))
    audio = b"".join(c.audio_int16_bytes for c in chunks)
    data = np.frombuffer(audio, dtype=np.int16)
    synth_time = time.perf_counter() - t0
    duration = len(data) / _piper_voice.config.sample_rate

    print(f"[{_ts()}][TTS] (synth {synth_time:.2f}s, audio {duration:.1f}s) {text}")

    _interrupted.clear()
    _is_speaking.set()
    _playback_done.clear()

    def _play_callback(outdata, frames, time_info, status):
        remaining = len(_play_callback.data) - _play_callback.pos
        if remaining <= 0 or _interrupted.is_set():
            outdata[:] = 0
            raise sd.CallbackStop
        n = min(frames, remaining)
        outdata[:n] = _play_callback.data[_play_callback.pos:_play_callback.pos + n]
        if n < frames:
            outdata[n:] = 0
        # Feed playback samples to AEC reference buffer (resample to mic rate)
        played = _play_callback.data[_play_callback.pos:_play_callback.pos + n, 0]
        if _piper_voice.config.sample_rate != _SAMPLE_RATE:
            ratio = _SAMPLE_RATE / _piper_voice.config.sample_rate
            indices = np.arange(0, len(played), 1 / ratio).astype(int)
            indices = indices[indices < len(played)]
            resampled = played[indices]
        else:
            resampled = played
        with _playback_lock:
            _playback_ring.extend(resampled.tolist())
        _play_callback.pos += n

    # Duplicate mono to stereo for proper speaker output
    stereo = np.column_stack([data, data])
    _play_callback.data = stereo
    _play_callback.pos = 0

    out_stream = sd.OutputStream(
        samplerate=_piper_voice.config.sample_rate, channels=2, dtype="int16",
        callback=_play_callback,
        finished_callback=lambda: _playback_done.set(),
        device=_OUTPUT_DEV,
    )
    out_stream.start()

    # Skip wake word check if Rosie is saying her own name
    _skip_wake = "rosie" in text.lower() or "rosy" in text.lower()

    if _DUPLEX:
        # During playback, check for wake word on main mic (production)
        # Also log debug mic (headset) for comparison
        check_interval = 0.5
        next_check = time.time() + check_interval
        while not _playback_done.is_set():
            if time.time() >= next_check:
                # Main mic — this is what triggers barge-in
                heard = ""
                with _mic_lock:
                    recent_chunks = [c for _, c in list(_mic_buffer)[-3:]]
                if recent_chunks:
                    raw_audio = np.concatenate(recent_chunks).astype(np.float32) / 32768.0
                    t0 = time.time()
                    result = _whisper_wake.transcribe(raw_audio, fp16=False, language="en")
                    heard = result["text"].strip().lower()
                    log = f"[{_ts()}][WAKE] ({time.time()-t0:.1f}s) mic:'{heard}'"

                    # Debug mic — just for logging comparison
                    if _DEBUG_MIC is not None:
                        with _debug_lock:
                            dbg_chunks = [c for _, c in list(_debug_buffer)[-3:]]
                        if dbg_chunks:
                            dbg_audio = np.concatenate(dbg_chunks).astype(np.float32) / 32768.0
                            dbg_result = _whisper_wake.transcribe(dbg_audio, fp16=False, language="en")
                            dbg_heard = dbg_result["text"].strip().lower()
                            log += f" | headset:'{dbg_heard}'"
                    print(log)

                if not _skip_wake and heard and ("rosie" in heard or "rosy" in heard or "rosi" in heard or "rose" in heard):
                    rosie_words = set(text.lower().split())
                    heard_words = set(heard.split())
                    new_words = heard_words - rosie_words - {"", "rosie", "rosy", "rosi"}
                    if new_words or heard.strip().rstrip(".,!") in ("rosie", "rosy", "rosi", "hey rosie", "hey rosy"):
                        print(f"[{_ts()}][TTS] Wake word detected: '{heard}'")
                        _interrupted.set()
                        break
                next_check = time.time() + check_interval
            time.sleep(0.05)
    else:
        # Check for stop during playback
        while not _playback_done.is_set():
            if stop_check and stop_check():
                _interrupted.set()
                print(f"[{_ts()}][TTS] Stopped by user")
                break
            time.sleep(0.05)

    out_stream.stop()
    out_stream.close()
    _is_speaking.clear()
    print(f"[{_ts()}][TTS] Playback done")


def interrupt():
    _interrupted.set()


def is_speaking() -> bool:
    return _is_speaking.is_set()


def get_listen_mode() -> str:
    return _listen_mode

def set_listen_mode(mode: str):
    global _listen_mode
    if mode in ("always", "name", "muted"):
        _listen_mode = mode
        _speech_event.clear()
        _wake_detected.clear()
        _oww_model.reset()
