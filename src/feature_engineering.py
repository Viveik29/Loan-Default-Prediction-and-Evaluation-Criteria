from sklearn.model_selection import train_test_split
import os
import pandas as pd
import yaml

def params_load(file_path):
    with open(file_path, "r") as f:
        params = yaml.safe_load(f)
        return params

def data_split(df):
    X = df.drop(columns=["Default"])
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def main():
    params = 'params.yaml'
    file_path="RAW_DATA\df_clean.csv"
    df = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = data_split(df)

    save_dir = os.path.join("RAW_DATA", "Clearn_data")
    os.makedirs(save_dir, exist_ok=True)

    file_paths1 = os.path.join(save_dir, "X_train.csv")
    file_paths2 = os.path.join(save_dir, "X_test.csv")
    file_paths3 = os.path.join(save_dir, "y_train.csv")
    file_paths4 = os.path.join(save_dir, "y_test.csv")

    
    # Save files
    X_train.to_csv(file_paths1, index=False)
    X_test.to_csv(file_paths2, index=False)
    y_train.to_csv(file_paths3, index=False)
    y_test.to_csv(file_paths4, index=False)


if __name__ == "__main__":
    main()
