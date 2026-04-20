# Wake Word Training

Train a custom "Rosie" wake word model using [OpenWakeWord](https://github.com/dscripka/openwakeword).

## Prerequisites

- WSL (Linux) with Python 3.11
- NVIDIA GPU with CUDA support
- ~18GB disk space (negative features dataset)

## First-Time Setup

Already done — the `venv/`, datasets, and repos are in this directory. If starting fresh:

```bash
cd train_wakeword
python3.11 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pyyaml tqdm datasets==2.14.6 pyarrow==15.0.2 mutagen torchinfo torchmetrics \
  audiomentations acoustics pronouncing webrtcvad piper-phonemize onnx speechbrain \
  torch_audiomentations torchaudio==2.5.1 espeak-phonemizer torchcodec -e ./openwakeword
sudo apt install espeak-ng git-lfs

# Clone repos
git clone https://github.com/dscripka/openwakeword.git
git clone https://github.com/dscripka/piper-sample-generator.git

# Download resource models
mkdir -p openwakeword/openwakeword/resources/models
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O openwakeword/openwakeword/resources/models/embedding_model.onnx
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O openwakeword/openwakeword/resources/models/melspectrogram.onnx
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O openwakeword/openwakeword/resources/models/embedding_model.tflite
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O openwakeword/openwakeword/resources/models/melspectrogram.tflite

# Download training data
wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy -O negative_features.npy
wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy -O validation_features.npy
```

## Training (3 Steps)

```bash
cd train_wakeword
source venv/bin/activate

# Step 1: Generate synthetic clips (~10 min)
python3 openwakeword/openwakeword/train.py --training_config rosie_model.yaml --generate_clips

# Step 2: Augment clips (~4 min)
python3 openwakeword/openwakeword/train.py --training_config rosie_model.yaml --augment_clips

# Step 3: Train model (~3 min)
python3 openwakeword/openwakeword/train.py --training_config rosie_model.yaml --train_model
```

Output: `my_custom_model/rosie.onnx`

## Deploy

```bash
cp my_custom_model/rosie.onnx ../models/rosie.onnx
```

## Adding Real Voice Recordings

To improve accuracy with Lynne's actual voice:

1. Record 50-100 clips of Lynne saying "Rosie" at different distances and tones
2. Save as 16kHz 16-bit WAV files
3. Place in `my_custom_model/positive_train/` alongside the synthetic clips
4. Increase `n_samples` and `steps` in `rosie_model.yaml`
5. Re-run Steps 2 and 3

## Config (rosie_model.yaml)

| Setting | Current | Notes |
|---|---|---|
| `n_samples` | 3000 | More = better accuracy, slower training |
| `n_samples_val` | 500 | Validation set size |
| `steps` | 15000 | Training steps |
| `target_phrase` | "rosie" | The wake word |

## Current Model Performance

- Accuracy: 65.6%
- Recall: 31.8%
- False positives/hour: 1.24

These numbers are from a small training run. Increasing samples to 10,000+ and steps to 50,000 should improve significantly.
