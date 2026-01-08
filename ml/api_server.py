"""
MUDRA DETECTION API SERVER
Flask API server that wraps the hybrid ML model for web integration.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from hybrid_webcam import (
    model, model_classes,
    get_scale_ref,
    detect_mudra_hybrid,
    RULE_MUDRA_FUNCTIONS,
    ML_CONF_THRESHOLD
)

app = Flask(__name__)
CORS(app)

print(f"ML Model loaded with {len(model_classes)} classes")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

SUPPORTED_MUDRAS = sorted(set(list(RULE_MUDRA_FUNCTIONS.keys()) + list(model_classes)))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "mudra_count": len(SUPPORTED_MUDRAS),
        "rule_based_mudras": len(RULE_MUDRA_FUNCTIONS),
        "ml_mudras": len(model_classes)
    }), 200

@app.route('/mudras', methods=['GET'])
def get_mudras():
    return jsonify({
        "mudras": SUPPORTED_MUDRAS,
        "count": len(SUPPORTED_MUDRAS),
        "rule_based": list(RULE_MUDRA_FUNCTIONS.keys()),
        "ml_based": model_classes.tolist()
    }), 200

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({
                "success": True,
                "hand_detected": False,
                "mudra": "No hand detected",
                "confidence": 0.0,
                "method": "NONE"
            }), 200
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        handedness = results.multi_handedness[0] if results.multi_handedness else None
        handedness_label = handedness.classification[0].label if handedness else "Right"
        
        mudra_name, confidence, method = detect_mudra_hybrid(
            landmarks, 
            handedness_label, 
            prev_landmarks=None
        )
        
        if mudra_name == "Stabilizing...":
            mudra_name = "Unknown"
            confidence = 0.0
            method = "NONE"
        
        if method is None:
            method = "NONE"
        
        response_data = {
            "success": True,
            "hand_detected": True,
            "mudra": mudra_name,
            "confidence": float(confidence),
            "method": method
        }
        
        if data.get('include_landmarks', False):
            response_data['landmarks'] = [[lm.x, lm.y, lm.z] for lm in landmarks]
        
        return jsonify(response_data), 200
        
    except Exception as e:
        import traceback
        print(f"Error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("MUDRA DETECTION API SERVER")
    print("=" * 60)
    print(f"Loaded {len(SUPPORTED_MUDRAS)} mudra classes")
    print(f"Rule-based: {len(RULE_MUDRA_FUNCTIONS)}, ML: {len(model_classes)}")
    print(f"Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
