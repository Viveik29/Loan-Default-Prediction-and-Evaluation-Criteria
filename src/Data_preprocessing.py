
import pandas as pd
import numpy as np
import os
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from scipy.stats import skew



def params_load(file_path):
    with open(file_path, "r") as f:
        params = yaml.safe_load(f)
        return params



# ----------------------------------------------------------------
# 1) Identify numeric & categorical columns
# ----------------------------------------------------------------
def get_columns(df, target="Default"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove target from lists
    if target in numeric_cols: numeric_cols.remove(target)
    if target in categorical_cols: categorical_cols.remove(target)

    return numeric_cols, categorical_cols

# ----------------------------------------------------------------
# 2) Drop high-missing columns
# ----------------------------------------------------------------
def drop_high_missing(df, threshold=0.60):
    missing_percent = df.isnull().mean()
    drop_cols = missing_percent[missing_percent > threshold].index.tolist()

    print("Dropping high-missing columns:", drop_cols)
    df = df.drop(columns=drop_cols)
    return df, drop_cols

# ----------------------------------------------------------------
# 3) Numeric Missing: Median Imputation
# ----------------------------------------------------------------

def impute_numeric(df, numeric_cols):
    if len(numeric_cols) == 0:
        return df, None

    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df, imputer

# ----------------------------------------------------------------
# 4) Categorical Missing: Fill with Mode or Constant
# ----------------------------------------------------------------
def impute_categorical(df, categorical_cols, fill_value="Unknown"):
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if df[col].mode().size else fill_value)
    return df

# ----------------------------------------------------------------
# 5) Detect and treat outliers - Winsorization (IQR method)
# ----------------------------------------------------------------
def winsorize_series(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return np.clip(s, lower, upper)


def winsorize_outliers(df, numeric_cols):
    if len(numeric_cols) == 0:
        return df
    for col in numeric_cols:
        df[col] = winsorize_series(df[col])
    return df

# ----------------------------------------------------------------
# 6) Log-transform highly skewed features
# ----------------------------------------------------------------

def log_transform(df, numeric_cols, threshold=1.0):
    if len(numeric_cols) == 0:
        return df

    for col in numeric_cols:
        if (df[col] > 0).all() and abs(skew(df[col])) > threshold:
            df[col] = np.log1p(df[col])
            print("Log-transformed:", col)
    return df


# ----------------------------------------------------------------
# 7) Encode Categorical Variables
# ----------------------------------------------------------------

def encode_categoricals(df, categorical_cols, target, cardinality_threshold=10):
    if len(categorical_cols) == 0:
        return df, {}, None

    low_cardinality = [c for c in categorical_cols if df[c].nunique() <= cardinality_threshold]
    high_cardinality = [c for c in categorical_cols if df[c].nunique() > cardinality_threshold]

    print("Low-cardinality Label Encoded:", low_cardinality)
    print("High-cardinality Target Encoded:", high_cardinality)

    label_encoders = {}

    # Label encoding for low-cardinality features
    for col in low_cardinality:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Target encoding only if there are columns
    target_encoder = None
    if len(high_cardinality) > 0:
        target_encoder = TargetEncoder(cols=high_cardinality)
        df[high_cardinality] = target_encoder.fit_transform(df[high_cardinality], df[target])

    return df, label_encoders, target_encoder


# ----------------------------------------------------------------
# 8) Full Preprocessing Pipeline
# ----------------------------------------------------------------
def preprocess_data(df, target="Default"):
    #df = df.copy()

    print("\nSTARTING PREPROCESSING...\n")

    # 1. Identify columns
    numeric_cols, categorical_cols = get_columns(df, target)

    # 2. Drop high-missing columns
    df, dropped_cols = drop_high_missing(df)

    # Update column lists after drops
    numeric_cols, categorical_cols = get_columns(df, target)
    

    # 3. Handle missing values
    df, num_imputer = impute_numeric(df, numeric_cols)
    df = impute_categorical(df, categorical_cols)

    # 4. Outlier treatment
    df = winsorize_outliers(df, numeric_cols)

    # 5. Apply log transform to skewed features
    df = log_transform(df, numeric_cols)

    # 6. Encode categorical variables
    df, label_encoders, target_encoder = encode_categoricals(
        df, categorical_cols, target
    )

    print("\nPREPROCESSING COMPLETED!\n")

    return df, {
        "numeric_imputer": num_imputer,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "dropped_columns": dropped_cols
    }

def main():
    params_path= 'params.yaml'
    params = params_load(params_path)
    file_paths = 'RAW_DATA/Raw/df_train.csv'
    df = pd.read_csv(file_paths)
    df_clean, preprocessors = preprocess_data(df, target="Default")
    print(df_clean.isnull().sum())
    #remove_stopwords = params["data_preprocessing"]["remove_stopwords"]
    #base_dir = r"D:\\Data Science case study\\-Loan-Default-Prediction-and-Evaluation-Criteria\\"
    os.makedirs("RAW_DATA",exist_ok=True)
    file_path = os.path.join("RAW_DATA", "df_clean.csv")
    # df_clean = pd.read_csv(file_path)
    # print(text_train)
    df_clean.to_csv(file_path,index=False)

if __name__ == '__main__':
    main()