import pytest
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
import sys

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ✅ ADD PATH FIRST (CRITICAL)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# ✅ SINGLE, CLEAN IMPORT STYLE
from Data_preprocessing import (
    winsorize_array,
    log_transform_array,
    build_pipeline,
    save_pickle,
    main,
    TARGET_COL
)


# -----------------------------
# Winsorization Tests
# -----------------------------
class TestWinsorizeArray:

    def test_winsorize_array_handles_outliers(self):
        data = np.array([1, 2, 3, 4, 5, 100]).reshape(-1, 1)
        winsorized = winsorize_array(data)

        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        assert np.max(winsorized) <= upper_bound

    def test_winsorize_array_preserves_shape(self):
        data = np.random.rand(10, 3)
        assert winsorize_array(data).shape == data.shape


# -----------------------------
# Log Transform Tests
# -----------------------------
class TestLogTransformArray:

    def test_log_transform_positive_values(self):
        data = np.array([[1, 10, 100], [2, 20, 200]]).T
        transformed = log_transform_array(data)

        assert transformed.shape == data.shape
        assert np.all(np.isfinite(transformed))

    def test_log_transform_negative_values(self):
        data = np.array([[1, 2, 3], [-1, -2, -3]]).T
        transformed = log_transform_array(data)

        assert transformed.shape == data.shape


# -----------------------------
# Pipeline Tests
# -----------------------------
class TestBuildPipeline:

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "age": [25, 30, 35, None],
            "income": [50000, 60000, None, 80000],
            "credit_score": [650, 700, 720, 680],
            "employment_type": ["A", "B", "A", "B"],
            "Default": [0, 1, 0, 1]
        })

    def test_pipeline_output_types(self, sample_data):
        preprocessor, features = build_pipeline(sample_data)

        assert isinstance(preprocessor, ColumnTransformer)
        assert isinstance(features, list)
        assert TARGET_COL not in features

    def test_pipeline_fit_transform(self, sample_data):
        preprocessor, features = build_pipeline(sample_data)

        preprocessor.fit(sample_data[features], sample_data[TARGET_COL])
        output = preprocessor.transform(sample_data[features])

        assert output.shape[0] == len(sample_data)
        assert not np.any(np.isnan(output))


# -----------------------------
# Pickle Tests
# -----------------------------
class TestSavePickle:

    def test_pickle_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pkl")
            obj = {"a": 1}

            save_pickle(obj, path)

            with open(path, "rb") as f:
                loaded = pickle.load(f)

            assert loaded == obj


# -----------------------------
# Main Function Test
# -----------------------------
class TestMain:

    def test_main_creates_files(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "RAW_DATA", "Raw"), exist_ok=True)
            os.makedirs(os.path.join(tmp, "Models"), exist_ok=True)

            df = pd.DataFrame({
                "age": [25, 30],
                "income": [50000, 60000],
                "Default": [0, 1]
            })

            df.to_csv(os.path.join(tmp, "RAW_DATA", "Raw", "df_train.csv"), index=False)

            monkeypatch.chdir(tmp)

            main()

            assert os.path.exists("Models/preprocessor.pkl")
            assert os.path.exists("RAW_DATA/df_clean.csv")
