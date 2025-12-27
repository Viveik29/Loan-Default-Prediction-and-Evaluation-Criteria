import pandas as pd
import numpy as np
import os
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder
from scipy.stats import skew


def params_load(file_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        print(f"Failed to load params from {file_path}: {e}")
        raise


def get_columns(df: pd.DataFrame, target: str = "Default") -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns excluding the target column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    return numeric_cols, categorical_cols


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.60) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with missing value ratio above the threshold."""
    missing_percent = df.isnull().mean()
    drop_cols = missing_percent[missing_percent > threshold].index.tolist()

    if drop_cols:
        print("Dropping high-missing columns:", drop_cols)
    df = df.drop(columns=drop_cols)
    return df, drop_cols


def impute_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, SimpleImputer | None]:
    """Impute missing numeric columns using median strategy."""
    if not numeric_cols:
        return df, None

    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df, imputer


def impute_categorical(df: pd.DataFrame, categorical_cols: list[str], fill_value: str = "Unknown") -> pd.DataFrame:
    """Impute missing categorical columns with mode or specified fill value."""
    for col in categorical_cols:
        fill = df[col].mode()[0] if not df[col].mode().empty else fill_value
        df[col] = df[col].fillna(fill)
    return df


def winsorize_series(s: pd.Series) -> pd.Series:
    """Clip outliers in a series using the IQR method (winsorization)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return np.clip(s, lower, upper)


def winsorize_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Apply winsorization to numeric columns."""
    for col in numeric_cols:
        df[col] = winsorize_series(df[col])
    return df


def log_transform(df: pd.DataFrame, numeric_cols: list[str], threshold: float = 1.0) -> pd.DataFrame:
    """Log-transform numeric columns with skewness above a threshold."""
    for col in numeric_cols:
        if (df[col] > 0).all() and abs(skew(df[col])) > threshold:
            df[col] = np.log1p(df[col])
            print(f"Log-transformed column: {col}")
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target: str,
    cardinality_threshold: int = 10
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], TargetEncoder | None]:
    """Encode categorical variables with label encoding or target encoding."""
    low_cardinality = [c for c in categorical_cols if df[c].nunique() <= cardinality_threshold]
    high_cardinality = [c for c in categorical_cols if df[c].nunique() > cardinality_threshold]

    if low_cardinality:
        print("Low-cardinality Label Encoded:", low_cardinality)
    if high_cardinality:
        print("High-cardinality Target Encoded:", high_cardinality)

    label_encoders = {}

    # Label encoding for low-cardinality columns
    for col in low_cardinality:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_encoder = None
    # Target encoding for high-cardinality columns
    if high_cardinality:
        target_encoder = TargetEncoder(cols=high_cardinality)
        df[high_cardinality] = target_encoder.fit_transform(df[high_cardinality], df[target])

    return df, label_encoders, target_encoder


def preprocess_data(df: pd.DataFrame, target: str = "Default") -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline:
    - Identify columns
    - Drop high missing columns
    - Impute missing values
    - Winsorize outliers
    - Log-transform skewed numeric columns
    - Encode categorical columns
    """
    df = df.copy()

    print("\nSTARTING PREPROCESSING...\n")

    # Identify numeric and categorical columns
    numeric_cols, categorical_cols = get_columns(df, target)

    # Drop columns with high missing values
    df, dropped_cols = drop_high_missing(df)

    # Re-identify columns after dropping
    numeric_cols, categorical_cols = get_columns(df, target)

    # Impute missing values
    df, num_imputer = impute_numeric(df, numeric_cols)
    df = impute_categorical(df, categorical_cols)

    # Treat outliers with winsorization
    df = winsorize_outliers(df, numeric_cols)

    # Log-transform highly skewed columns
    df = log_transform(df, numeric_cols)

    # Encode categorical columns
    df, label_encoders, target_encoder = encode_categoricals(df, categorical_cols, target)

    print("\nPREPROCESSING COMPLETED!\n")

    preprocessors = {
        "numeric_imputer": num_imputer,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "dropped_columns": dropped_cols
    }
    return df, preprocessors


def main():
    """Main function to run preprocessing on training data."""
    params_path = 'params.yaml'
    params = params_load(params_path)

    input_path = 'RAW_DATA/Raw/df_train.csv'
    df = pd.read_csv(input_path)

    df_clean, preprocessors = preprocess_data(df, target="Default")
    print(df_clean.isnull().sum())

    os.makedirs("RAW_DATA", exist_ok=True)
    output_path = os.path.join("RAW_DATA", "df_clean.csv")
    df_clean.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
