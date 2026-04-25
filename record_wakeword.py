"""Record 30 'Rosie' wake word samples via Plantronics headset."""
import sounddevice as sd
import numpy as np
import wave, os, time

DEVICE = 1  # Plantronics headset
RATE = 16000
DURATION = 2.0
OUT_DIR = "train_wakeword/real_samples"
os.makedirs(OUT_DIR, exist_ok=True)

prompts = [
    "1/30  - Normal: 'Rosie'",
    "2/30  - Normal: 'Rosie'",
    "3/30  - Normal: 'Rosie'",
    "4/30  - Normal: 'Rosie'",
    "5/30  - Normal: 'Rosie'",
    "6/30  - Normal: 'Hey Rosie'",
    "7/30  - Normal: 'Hey Rosie'",
    "8/30  - Normal: 'Hey Rosie'",
    "9/30  - Normal: 'Hi Rosie'",
    "10/30 - Normal: 'Hi Rosie'",
    "11/30 - LOUD: 'Rosie!'",
    "12/30 - LOUD: 'Rosie!'",
    "13/30 - LOUD: 'Hey Rosie!' (like calling from another room)",
    "14/30 - LOUD: 'Rosie!'",
    "15/30 - LOUD: 'Hey Rosie!'",
    "16/30 - Quiet: 'Rosie' (whisper)",
    "17/30 - Quiet: 'Rosie'",
    "18/30 - Quiet: 'Hey Rosie'",
    "19/30 - Quiet: 'Rosie'",
    "20/30 - Quiet: 'Hey Rosie'",
    "21/30 - Question: 'Rosie?'",
    "22/30 - Question: 'Rosie?'",
    "23/30 - Question: 'Hey Rosie?'",
    "24/30 - Question: 'Rosie, are you there?'",
    "25/30 - Question: 'Rosie?'",
    "26/30 - Casual: 'Rosie' (fast, mumbled)",
    "27/30 - Casual: 'Roooosie' (drawn out)",
    "28/30 - Casual: 'Hey Rosie' (fast)",
    "29/30 - Casual: 'Rosie' (mid-sentence tone)",
    "30/30 - Casual: 'Rosie' (like you just woke up)",
]

print(f"Recording {len(prompts)} samples on device {DEVICE} at {RATE}Hz")
print("Press Enter for each recording. Type 'q' to quit.\n")

for i, prompt in enumerate(prompts):
    print(f"  {prompt}")
    resp = input("  Press Enter to record... ")
    if resp.strip().lower() == "q":
        print("Stopped early.")
        break

    audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1,
                   dtype="int16", device=DEVICE)
    print("  * Recording...", end="", flush=True)
    sd.wait()
    print(" done!")

    path = os.path.join(OUT_DIR, f"rosie_{i+1:02d}.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())
    print(f"  Saved: {path}\n")

print(f"\nAll done! {i+1} samples saved to {OUT_DIR}/")
