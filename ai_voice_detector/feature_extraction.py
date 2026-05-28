"""
Extract audio features that help distinguish AI-generated speech from real human speech.

Key differences between AI and human audio:
- AI audio tends to have smoother spectral envelopes
- AI lacks natural micro-variations in pitch (jitter) and amplitude (shimmer)
- AI often has unnatural breathing patterns or none at all
- AI has more consistent background noise profiles
- AI pitch transitions are smoother (less natural wobble)
"""

import numpy as np
import librosa


def extract_features(audio_path: str, sr: int = 16000) -> dict:
    """
    Extract a feature vector from an audio file for AI vs human classification.

    Returns a dict of features that can be flattened into a vector.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Trim silence from edges
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trimmed) < sr * 0.5:
        y_trimmed = y  # fallback if too short after trim

    features = {}

    # --- Spectral features ---
    # MFCCs (mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i}_delta_std'] = np.std(np.diff(mfccs[i]))

    # Spectral centroid (brightness)
    spec_cent = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_std'] = np.std(spec_cent)

    # Spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_std'] = np.std(spec_bw)

    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    features['spectral_rolloff_std'] = np.std(spec_rolloff)

    # Spectral flatness (how noise-like vs tonal)
    spec_flat = librosa.feature.spectral_flatness(y=y_trimmed)[0]
    features['spectral_flatness_mean'] = np.mean(spec_flat)
    features['spectral_flatness_std'] = np.std(spec_flat)

    # --- Pitch/F0 features ---
    f0, voiced_flag, _ = librosa.pyin(
        y_trimmed, fmin=50, fmax=500, sr=sr
    )
    f0_valid = f0[~np.isnan(f0)]

    if len(f0_valid) > 1:
        features['f0_mean'] = np.mean(f0_valid)
        features['f0_std'] = np.std(f0_valid)
        features['f0_range'] = np.ptp(f0_valid)
        # Jitter (pitch perturbation) - key AI indicator
        jitter = np.mean(np.abs(np.diff(f0_valid))) / np.mean(f0_valid)
        features['jitter'] = jitter
        # Pitch slope variability
        features['f0_diff_std'] = np.std(np.diff(f0_valid))
    else:
        features['f0_mean'] = 0
        features['f0_std'] = 0
        features['f0_range'] = 0
        features['jitter'] = 0
        features['f0_diff_std'] = 0

    # Voiced ratio
    features['voiced_ratio'] = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0

    # --- Energy/amplitude features ---
    rms = librosa.feature.rms(y=y_trimmed)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    # Shimmer (amplitude perturbation) - another AI indicator
    if len(rms) > 1:
        shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
        features['shimmer'] = shimmer
    else:
        features['shimmer'] = 0

    # --- Temporal features ---
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # Silence ratio (indicator of natural pauses)
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(y_trimmed) < silence_threshold) / len(y_trimmed)
    features['silence_ratio'] = silence_ratio

    # --- Harmonic features ---
    harmonic, percussive = librosa.effects.hpss(y_trimmed)
    features['harmonic_mean'] = np.mean(np.abs(harmonic))
    features['percussive_mean'] = np.mean(np.abs(percussive))
    features['harmonic_percussive_ratio'] = (
        np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-8)
    )

    return features


def features_to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to a sorted, consistent numpy vector."""
    return np.array([features[k] for k in sorted(features.keys())])


def get_feature_names() -> list:
    """Get sorted list of feature names (for reference)."""
    # Generate a dummy to get keys
    # In practice, call extract_features on any file and get sorted keys
    return sorted([
        *[f'mfcc_{i}_mean' for i in range(20)],
        *[f'mfcc_{i}_std' for i in range(20)],
        *[f'mfcc_{i}_delta_std' for i in range(20)],
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std',
        'spectral_flatness_mean', 'spectral_flatness_std',
        'f0_mean', 'f0_std', 'f0_range', 'jitter', 'f0_diff_std',
        'voiced_ratio',
        'rms_mean', 'rms_std', 'shimmer',
        'zcr_mean', 'zcr_std',
        'silence_ratio',
        'harmonic_mean', 'percussive_mean', 'harmonic_percussive_ratio',
    ])
