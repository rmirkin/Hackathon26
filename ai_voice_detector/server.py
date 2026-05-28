"""
Flask API server for the AI voice detector.

Endpoints:
  POST /analyze  - Upload an audio file, get AI vs Human prediction
  GET  /health   - Health check
"""

import os
import pickle
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from feature_extraction import extract_features, features_to_vector


app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "detector.pkl")

# Load model at startup
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
else:
    print(f"WARNING: No model found at {MODEL_PATH}. Run train_model.py first.")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze an uploaded audio file.

    Expects multipart form data with an 'audio' file field.
    Returns JSON with prediction and confidence.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided. Send as "audio" field.'}), 400

    audio_file = request.files['audio']

    # Save to temp file for processing
    suffix = os.path.splitext(audio_file.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Extract features
        features = extract_features(tmp_path)
        feature_vector = features_to_vector(features).reshape(1, -1)

        # Predict
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]

        # Get confidence
        ai_confidence = float(probabilities[1])
        human_confidence = float(probabilities[0])

        result = {
            'prediction': 'AI-Generated' if prediction == 1 else 'Human',
            'is_ai': bool(prediction == 1),
            'confidence': {
                'ai': round(ai_confidence * 100, 1),
                'human': round(human_confidence * 100, 1),
            },
            'features_analyzed': len(features),
            # Include some interpretable features
            'indicators': {
                'jitter': round(features.get('jitter', 0), 6),
                'shimmer': round(features.get('shimmer', 0), 6),
                'f0_std': round(features.get('f0_std', 0), 2),
                'spectral_flatness': round(features.get('spectral_flatness_mean', 0), 6),
                'silence_ratio': round(features.get('silence_ratio', 0), 4),
                'harmonic_percussive_ratio': round(features.get('harmonic_percussive_ratio', 0), 4),
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
