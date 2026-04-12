# Rosie — Budget AI Companion for Older Adults

A budget-friendly, open-source alternative to the [ElliQ](https://elliq.com/) companion robot. Built with off-the-shelf hardware and local AI, it provides **proactive companionship**, medication reminders, video calls, and a digital photo frame — all without a monthly subscription.

## Why This Exists

ElliQ costs ~$600 upfront + $40–$60/month. This project replicates its core value — a warm, proactive AI companion that checks in on older adults living alone — for under $150 in hardware and $0 in recurring costs.

## Features

| Feature | How It Works |
|---|---|
| **Voice Companion** | Local LLM with a warm, encouraging persona via Ollama |
| **Proactive Check-ins** | Scheduled prompts ("Have you had water today?") |
| **Caregiver Notes** | Family can leave voice or text notes; Rosie delivers them naturally |
| **User Profile** | Personalized — knows the user's name, hobbies, family |
| **Persistent Memory** | Remembers details across sessions |
| **Web UI** | Robot face with status indicators, clock, chat transcript |
| **Text & Voice Input** | Type or speak — both work |
| **Stop Button** | Interrupt Rosie mid-sentence |
| **Medication Reminders** | Configurable alerts via text-to-speech |

## Architecture

```
User speaks ──► Whisper (STT) ──► Ollama LLM ──► Piper (TTS) ──► Speaker
                                       ▲
Scheduler / Caregiver Notes ───────────┘  (proactive triggers)
         ▲
Flask Web UI ──► Status, Transcript, Controls, Caregiver Panel
```

All AI runs locally on the device. No cloud dependency, no data leaves the home.

## Project Structure

```
rosie/
├── main.py              # Entry point: voice loop + proactive + web UI
├── brain.py             # LLM interaction via Ollama, memory extraction
├── voice.py             # STT (Whisper) + TTS (Piper) + mic management
├── memory.py            # Persistent memory across sessions
├── proactive.py         # Scheduled check-ins & reminders
├── aec.py               # Acoustic echo cancellation module
├── config.yaml          # Persona, schedules, reminders, voice settings
├── profile.yaml         # User profile (name, hobbies, health, favorites)
├── caregivers.yaml      # Caregiver profiles (name, relationship, pronouns)
├── requirements.txt     # Python dependencies
└── ui/
    ├── app.py           # Flask backend (state, caregiver notes, controls)
    ├── static/app.js    # Shared UI logic
    └── templates/themes/
        ├── default.html # Simple theme
        └── robot.html   # Robot theme (default)
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed with a model pulled (`ollama pull gemma2:9b`)
- A working microphone and speaker
- [ffmpeg](https://ffmpeg.org/) installed (required by Whisper)
  - Windows: `winget install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Mac: `brew install ffmpeg`

### Install

```bash
git clone https://github.com/spencer-ok/ai-assistant.git
cd ai-assistant
python -m venv winvenv
winvenv\Scripts\activate        # Windows
# source winvenv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Download a Piper voice model:
```bash
mkdir models
cd models
curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
cd ..
```

### Find Audio Devices

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Note the device IDs for your microphone (input) and speakers (output), then update `config.yaml`.

### Configure

Edit `profile.yaml` with the user's info:
```yaml
user:
  name: "Lynne"
  gender: "female"
  pronouns: "she/her"
  hobbies: ["puzzles"]
```

Edit `caregivers.yaml` with family/caregiver info:
```yaml
caregivers:
  - name: "Val"
    relationship: "spouse"
    gender: "male"
    pronouns: "he/him"
    notes: "Primary caregiver, lives in the same house"
```

Edit `config.yaml` to set audio devices and schedules.

### Run

```bash
python main.py
```

Open `http://localhost:5000` for the web UI.

## Web UI

- **Robot face** with animated eyes and LED mouth
- **Large clock** with date — always visible
- **Status indicators** — green background (listening), red (thinking/speaking)
- **Chat transcript** — shows conversation history in real-time
- **Text input** — type messages as alternative to voice
- **Stop button** — interrupt Rosie mid-sentence
- **Mic mute** — toggle microphone on/off
- **Caregiver Notes panel** — leave text or voice notes for the user
- **Theme support** — `?theme=default` or `?theme=robot`

## Activities

Rosie can do interactive activities with the user. Just say the trigger word to start.

| Activity | Trigger Words | Description |
|---|---|---|
| **Trivia** | "trivia", "quiz me" | Multi-category trivia with hints and scoring |
| **Stories** | "tell me a story" | Full-length stories told naturally |
| **Jokes** | "tell me a joke" | Clean, family-friendly jokes |
| **Reminiscing** | "remember when" | Talk about memories and the past |
| **Word Games** | "word game" | Simple guessing games |

### Trivia System

Trivia uses a layered selection flow:
1. User says "let's do trivia"
2. Rosie offers categories (TV Shows, General Knowledge)
3. User picks a category → Rosie offers specific topics
4. User picks a topic → questions load from a YAML file
5. Questions are shuffled and asked one at a time with hints

Adding new trivia topics is easy — just drop a YAML file in `activities/trivia/`:

```yaml
topic: "My Topic"
instructions: >
  Quiz instructions for the LLM...
questions:
  - q: "Question text"
    a: "Answer"
    hint: "Hint text"
    difficulty: easy
```

Then register it in `activities/activities.yaml` under the appropriate category.

## Caregiver Notes

Family members can leave notes through the web UI. Notes are:
- Delivered when the user next speaks to Rosie
- Spoken aloud with a personal greeting ("Hello Lynne! Val left you a message...")
- Stored in Rosie's context so she can answer follow-up questions ("Where's Val?")
- Recordable via voice (Record Note button) or typed

## Configuration

### config.yaml

```yaml
persona:
  name: "Rosie"
  model: "gemma2:9b"
  system_prompt: >
    You are Rosie, a kind and caring companion...

voice:
  piper_model: "models/en_US-lessac-medium.onnx"
  whisper_model: "base"
  silence_duration: 1.3
  duplex: false
  input_device: 2      # Mic device ID
  output_device: 5     # Speaker device ID

schedule:
  - time: "09:00"
    message: "Good morning! Have you had breakfast yet?"

reminders:
  - time: "08:00"
    message: "Time to take your morning medication."

ollama:
  host: "http://192.168.68.115:11434"       # Remote GPU server
  fallback_host: "http://localhost:11434"    # Local fallback
  fallback_model: "qwen2.5:3b"              # Smaller local model
```

### Network LLM

Rosie supports running the LLM on a remote machine with a better GPU. If the remote server is unreachable, it automatically falls back to a local model.

```yaml
ollama:
  host: "http://<remote-ip>:11434"
  fallback_host: "http://localhost:11434"
  fallback_model: "qwen2.5:3b"
```

On the remote machine:
1. Install [Ollama](https://ollama.com/) and pull a model (`ollama pull gemma2:9b`)
2. Set environment variable `OLLAMA_HOST=0.0.0.0` and restart Ollama
3. Open firewall port 11434

## Hardware (~$125–$160)

| Component | Cost | Purpose |
|---|---|---|
| Raspberry Pi 4/5 (or any PC) | $45–$80 | Main compute |
| 7" Touchscreen | ~$50 | Face UI, clock, controls |
| USB Mic + Speaker | ~$20 | Voice I/O |
| Optional: ReSpeaker USB Mic Array | ~$35 | Better voice pickup with echo cancellation |

## Software Stack

| Layer | Tool | Role |
|---|---|---|
| LLM | [Ollama](https://ollama.com/) (gemma2:9b) | Conversation logic |
| Speech-to-Text | [OpenAI Whisper](https://github.com/openai/whisper) | Voice input |
| Text-to-Speech | [Piper](https://github.com/rhasspy/piper) | Natural voice output |
| UI | Flask | Web-based robot face and controls |
| Automation | Schedule library | Proactive check-ins |

## Development Phases

- [x] **Phase 1** — Voice loop: listen → think → speak
- [x] **Phase 1.5** — Performance & quality tuning
  - [x] Streaming LLM responses (sentence-by-sentence TTS)
  - [x] Auto-calibrating silence detection
  - [x] Whisper language hints for faster STT
  - [x] Persistent memory across sessions
  - [x] Time-aware persona
  - [x] Model optimization (qwen2.5:3b, 69% GPU)
- [x] **Phase 2** — Web UI with robot face, clock, controls
  - [x] Caregiver notes system
  - [x] User and caregiver profiles
  - [x] Theme support (default + robot)
  - [x] Responsive UI
  - [x] Network LLM with automatic fallback
- [ ] **Phase 3** — Sensor integration for proactive triggers
- [ ] **Phase 4** — Video calls & photo frame

## Backlog

- [ ] **Conversational setup mode** — A family member describes the user verbally and Rosie extracts/populates the profile automatically
- [ ] **Input method for older adults** — Solve the silence-timeout problem (wake word, tap-to-finish, visual feedback, or hybrid approach)
- [ ] **Mid-sentence barge-in with speaker output** — Wake word detection during TTS works with headphones but not reliably with open speakers. Hardware mic array (ReSpeaker) recommended.
- [ ] **LLM hallucination** — Small model fabricates facts. Larger model or cloud API would solve this.
- [ ] **Better mic hardware** — ReSpeaker USB mic array or similar for reliable voice pickup
- [ ] **Cloud LLM option** — Together AI API (~$0.50-$4/month) for deployments without a local GPU. OpenAI-compatible API, minimal code change.
- [ ] **Photo frame mode** — Background slideshow from shared cloud folder
- [ ] **Video calls** — One-click family calling via Jitsi Meet

## Source Control

Repository: [github.com/spencer-ok/ai-assistant](https://github.com/spencer-ok/ai-assistant)

```bash
# Push changes (PAT token already configured in git remote)
cd C:\apps\ai-assistant
git add -A
git commit -m "description of changes"
git push
```

## License

MIT
