
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import uuid
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load trained model
model = joblib.load("student_result_model.pkl")

FEATURES = [
    "internal_marks",
    "assignment_marks",
    "previous_score",
    "attendance"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "student-result-prediction",
        "status": "ok"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid JSON input"
            }), 400

        missing = [
            field for field in FEATURES
            if field not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        try:
            values = [float(data[field]) for field in FEATURES]
        except (ValueError, TypeError):
            return jsonify({
                "error": "All fields must contain numeric values"
            }), 400

        if not all(np.isfinite(values)):
            return jsonify({
                "error": "Values must be finite numbers"
            }), 400

        if any(value < 0 or value > 100 for value in values):
            return jsonify({
                "error": "Marks and attendance must be between 0 and 100"
            }), 400

        prediction = int(model.predict([values])[0])

        result = "PASS" if prediction == 1 else "FAIL"

        app.logger.info(
            "Request ID: %s | Prediction: %s | Duration: %.4f seconds",
            request_id,
            result,
            time.time() - start_time
        )

        return jsonify({
            "request_id": request_id,
            "prediction": result,
            "prediction_code": prediction
        })

    except Exception:
        app.logger.exception("Unexpected prediction error")
        return jsonify({
            "error": "Internal server error",
            "request_id": request_id
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
