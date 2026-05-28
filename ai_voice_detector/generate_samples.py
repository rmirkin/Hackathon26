"""
Generate fake AI voice samples using Bark (open-source TTS by Suno).
These will be used as "AI-generated" training/demo data.
"""

import os
import numpy as np
from scipy.io.wavfile import write as write_wav

# Bark generates very realistic speech
from bark import SAMPLE_RATE, generate_audio, preload_models


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "samples", "ai_generated")


SAMPLE_TEXTS = [
    "Hello, this is your bank calling. We've detected suspicious activity on your account and need to verify your identity.",
    "Hi there, I'm calling from tech support. We've noticed your computer has been sending us error reports.",
    "Good afternoon, this is the IRS. You have an outstanding tax balance that must be paid immediately to avoid legal action.",
    "Hey, it's me. I'm in trouble and I need you to wire me some money right away. Please don't tell anyone.",
    "This is an automated message from your insurance provider. Your policy is about to expire and requires immediate renewal.",
    "Hi, I'm calling about the car accident you were recently involved in. You may be entitled to compensation.",
    "Hello, congratulations! You've been selected for a special offer. Press one to claim your prize now.",
    "This is a courtesy call from your electric company. Your service will be disconnected unless payment is received today.",
]

# Different Bark speaker presets for variety
SPEAKERS = [
    "v2/en_speaker_0",
    "v2/en_speaker_1",
    "v2/en_speaker_2",
    "v2/en_speaker_3",
    "v2/en_speaker_6",
    "v2/en_speaker_9",
]


def generate_samples():
    """Generate AI voice samples using Bark."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Preloading Bark models (this may take a few minutes on first run)...")
    preload_models()

    print(f"\nGenerating {len(SAMPLE_TEXTS)} AI voice samples...")
    for i, text in enumerate(SAMPLE_TEXTS):
        speaker = SPEAKERS[i % len(SPEAKERS)]
        print(f"\n[{i+1}/{len(SAMPLE_TEXTS)}] Generating: \"{text[:50]}...\"")
        print(f"  Speaker: {speaker}")

        audio_array = generate_audio(text, history_prompt=speaker)

        # Normalize to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)

        filename = f"ai_sample_{i:03d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        write_wav(filepath, SAMPLE_RATE, audio_int16)
        print(f"  Saved: {filepath}")

    print(f"\nDone! Generated {len(SAMPLE_TEXTS)} samples in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_samples()
