# AI Voice Call Detector

Detects whether a phone call audio is AI-generated (deepfake/TTS) or a real human voice.

## How It Works

The detector extracts audio features that differ between AI-generated and human speech:
- **Jitter** — pitch perturbation (AI voices are unnaturally smooth)
- **Shimmer** — amplitude perturbation (AI lacks micro-variations)
- **MFCCs** — spectral envelope characteristics
- **F0 variation** — pitch dynamics (AI tends to be more monotone)
- **Silence patterns** — natural breathing/pauses vs AI's continuous output
- **Harmonic structure** — AI often has cleaner harmonics than real speech

## Setup

```bash
cd ai_voice_detector
pip install -r requirements.txt
```

## Usage

### 1. Generate AI voice samples (using Bark TTS)

```bash
python generate_samples.py
```

This creates realistic AI-generated speech samples in `samples/ai_generated/`.

### 2. Add real human samples

Place real human voice recordings (phone calls, voice memos) in:
```
samples/human/
```

Supported formats: .wav, .mp3, .flac, .ogg

### 3. Train the model

```bash
python train_model.py
```

### 4. Run the web demo

```bash
python server.py
```

Then open `index.html` in your browser (or serve it). The web UI lets you:
- Record audio from your microphone
- Upload an audio file
- Get a real-time AI vs Human verdict with confidence score

## Project Structure

```
ai_voice_detector/
├── generate_samples.py   # Generate fake AI voice samples with Bark
├── feature_extraction.py # Audio feature extraction pipeline
├── train_model.py        # Train the classifier
├── server.py             # Flask API server
├── index.html            # Web demo frontend
├── requirements.txt      # Python dependencies
├── samples/
│   ├── ai_generated/     # AI TTS samples (generated)
│   └── human/            # Real human recordings (you provide)
└── model/
    └── detector.pkl      # Trained model (after training)
```

## Demo Flow

1. Generate AI samples with `generate_samples.py`
2. Record yourself + friends saying similar phrases → put in `samples/human/`
3. Train with `train_model.py`
4. Start server, open the web UI
5. Play an AI-generated call → watch it get flagged as AI
6. Record yourself speaking → watch it get classified as Human
