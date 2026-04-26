"""Generate 'rosie' wake word clips from multiple Piper ONNX voices."""
import os, json, random, uuid, wave
import numpy as np
from scipy.signal import resample_poly
from math import gcd

try:
    import onnxruntime as ort
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "onnxruntime"], check=True)
    import onnxruntime as ort

from piper_phonemize import phonemize_espeak

MODELS_DIR = "piper-sample-generator/models"
OUTPUT_DIR = "my_custom_model/rosie/positive_train"
OUTPUT_DIR_TEST = "my_custom_model/rosie/positive_test"

VOICES = [
    "en_US-lessac-medium.onnx",   # female
    "en_US-joe-medium.onnx",      # male
    "en_US-ryan-medium.onnx",     # male
]

PHRASE = "rosie"
SAMPLES_PER_VOICE = 3000
TEST_SAMPLES_PER_VOICE = 500
LENGTH_SCALES = [0.75, 1.0, 1.25]
NOISE_SCALES = [0.667, 0.8, 1.0]
NOISE_W_SCALES = [0.8, 0.9, 1.0]


def phonemes_to_ids(phonemes, id_map):
    ids = [0]  # pad
    for p in phonemes:
        if p in id_map:
            ids.extend(id_map[p])
    ids.append(0)  # pad
    return ids


def generate_with_piper(model_path, output_dir, n_samples):
    config_path = model_path + ".json"
    with open(config_path) as f:
        config = json.load(f)

    num_speakers = config.get("num_speakers", 1)
    sample_rate = config["audio"]["sample_rate"]
    id_map = config["phoneme_id_map"]

    os.makedirs(output_dir, exist_ok=True)

    session = ort.InferenceSession(model_path)
    input_names = [i.name for i in session.get_inputs()]

    # Get phonemes for "rosie"
    phoneme_lists = phonemize_espeak(PHRASE, "en-us")

    print(f"  Generating {n_samples} clips with {os.path.basename(model_path)}...")

    generated = 0
    while generated < n_samples:
        length_scale = random.choice(LENGTH_SCALES)
        noise_scale = random.choice(NOISE_SCALES)
        noise_w = random.choice(NOISE_W_SCALES)
        speaker_id = random.randint(0, max(0, num_speakers - 1)) if num_speakers > 1 else None

        for phonemes in phoneme_lists:
            if generated >= n_samples:
                break

            ids = phonemes_to_ids(phonemes, id_map)
            ids_arr = np.array([ids], dtype=np.int64)
            ids_len = np.array([len(ids)], dtype=np.int64)
            scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

            inputs = {"input": ids_arr, "input_lengths": ids_len, "scales": scales}
            if "sid" in input_names and speaker_id is not None:
                inputs["sid"] = np.array([speaker_id], dtype=np.int64)

            audio = session.run(None, inputs)[0].squeeze()
            audio = np.clip(audio, -1.0, 1.0)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                g = gcd(sample_rate, 16000)
                audio = resample_poly(audio, 16000 // g, sample_rate // g)
                audio = np.clip(audio, -1.0, 1.0)

            audio_int16 = (audio * 32767).astype(np.int16)

            fname = uuid.uuid4().hex + ".wav"
            fpath = os.path.join(output_dir, fname)
            with wave.open(fpath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())

            generated += 1
            if generated % 500 == 0:
                print(f"    {generated}/{n_samples}")

    print(f"  Done: {generated} clips")


if __name__ == "__main__":
    for voice in VOICES:
        model_path = os.path.join(MODELS_DIR, voice)
        print(f"\n=== {voice} ===")
        generate_with_piper(model_path, OUTPUT_DIR, SAMPLES_PER_VOICE)
        generate_with_piper(model_path, OUTPUT_DIR_TEST, TEST_SAMPLES_PER_VOICE)

    train_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
    test_count = len([f for f in os.listdir(OUTPUT_DIR_TEST) if f.endswith('.wav')])
    print(f"\nTotal: {train_count} train, {test_count} test clips")
