"""
Train a classifier to distinguish AI-generated speech from real human speech.

Directory structure expected:
  samples/
    ai_generated/   <- AI TTS samples (from generate_samples.py)
    human/          <- Real human voice recordings

The trained model is saved to model/detector.pkl
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from feature_extraction import extract_features, features_to_vector


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_dataset():
    """Load audio files and extract features."""
    X = []
    y = []

    ai_dir = os.path.join(SAMPLES_DIR, "ai_generated")
    human_dir = os.path.join(SAMPLES_DIR, "human")

    # Load AI samples (label = 1)
    if os.path.exists(ai_dir):
        ai_files = [f for f in os.listdir(ai_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        print(f"Found {len(ai_files)} AI-generated samples")
        for fname in ai_files:
            filepath = os.path.join(ai_dir, fname)
            try:
                features = extract_features(filepath)
                X.append(features_to_vector(features))
                y.append(1)  # 1 = AI
            except Exception as e:
                print(f"  Error processing {fname}: {e}")
    else:
        print(f"WARNING: No AI samples directory found at {ai_dir}")

    # Load human samples (label = 0)
    if os.path.exists(human_dir):
        human_files = [f for f in os.listdir(human_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        print(f"Found {len(human_files)} human samples")
        for fname in human_files:
            filepath = os.path.join(human_dir, fname)
            try:
                features = extract_features(filepath)
                X.append(features_to_vector(features))
                y.append(0)  # 0 = human
            except Exception as e:
                print(f"  Error processing {fname}: {e}")
    else:
        print(f"WARNING: No human samples directory found at {human_dir}")

    return np.array(X), np.array(y)


def train():
    """Train the detector model."""
    print("Loading dataset and extracting features...")
    X, y = load_dataset()

    if len(X) == 0:
        print("\nNo samples found! Please add audio files to:")
        print(f"  AI samples:    {os.path.join(SAMPLES_DIR, 'ai_generated')}/")
        print(f"  Human samples: {os.path.join(SAMPLES_DIR, 'human')}/")
        return

    print(f"\nDataset: {len(X)} samples ({np.sum(y == 1)} AI, {np.sum(y == 0)} human)")
    print(f"Feature vector size: {X.shape[1]}")

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ))
    ])

    # Cross-validation
    if len(X) >= 10:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Train/test split for detailed report
    if len(X) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print("\nClassification Report (test set):")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    else:
        # Too few samples for split, just train on all
        pipeline.fit(X, y)
        print("\n(Too few samples for train/test split, trained on all data)")

    # Retrain on full dataset for deployment
    pipeline.fit(X, y)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "detector.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
