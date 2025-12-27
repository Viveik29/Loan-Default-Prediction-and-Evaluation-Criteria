from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

def create_app(testing=False):
    app = Flask(__name__)
    app.config["TESTING"] = testing

    model, scaler, label_encoders = None, None, None

    def init_model():
        nonlocal model, scaler, label_encoders

        model_path = "Model/model.pkl"
        preprocessing_path = "Model/preprocessing.pkl"

        if testing:
            # mock objects for tests
            class DummyModel:
                def predict(self, X): return [0]
                def predict_proba(self, X): return [[0.9, 0.1]]
                feature_importances_ = [0.1] * X.shape[1]

            model = DummyModel()
            scaler = lambda x: x
            label_encoders = {}
            return

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(preprocessing_path, "rb") as f:
            data = pickle.load(f)
            scaler = data["scaler"]
            label_encoders = data["label_encoders"]

    @app.route("/")
    def home():
        return "OK"

    @app.route("/predict", methods=["POST"])
    def predict():
        data = pd.DataFrame([request.form])
        prediction = model.predict(data)[0]
        return jsonify({"prediction": int(prediction)})

    init_model()
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=5001)
