import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "Models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PREPROCESS_PATH = os.path.join(MODEL_DIR, "preprocessing.pkl")

# -------------------------------------------------
# App Factory (KEY FIX)
# -------------------------------------------------
def create_app(testing: bool = False):
    app = Flask(__name__)
    app.config["TESTING"] = testing

    if not testing:
        load_model(app)

    # ---------------- Routes ----------------

    @app.route("/")
    def home():
        return "Loan Default Prediction API is running"

    @app.route("/predict", methods=["POST"])
    def predict():
        model = app.config.get("MODEL")
        scaler = app.config.get("SCALER")
        encoders = app.config.get("ENCODERS")

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        input_df = pd.DataFrame([request.json or request.form])

        # Convert numerics safely
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="ignore")

        # Encode categoricals
        for col, encoder in encoders.items():
            if col in input_df:
                input_df[col] = encoder.transform(input_df[col].astype(str))

        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        return jsonify({
            "prediction": int(pred),
            "probability": round(float(prob), 4)
        })

    return app

# -------------------------------------------------
# Model Loader (NO TRAINING IN CI)
# -------------------------------------------------
def load_model(app):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESS_PATH, "rb") as f:
        data = pickle.load(f)

    app.config["MODEL"] = model
    app.config["SCALER"] = data["scaler"]
    app.config["ENCODERS"] = data["label_encoders"]

# -------------------------------------------------
# Local Run Only
# -------------------------------------------------
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)
