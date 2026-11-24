import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.drop(columns=["id"], errors="ignore")

    categorical_cols = ["proto", "service", "state"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].replace("-", "unknown").fillna("unknown")

    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

    return df


def build_preprocessor(X_fit: pd.DataFrame):

    numerical_cols = X_fit.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = [
        col for col in ["proto", "service", "state"]
        if col in X_fit.columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_cols,
            ),
        ]
    )

    return preprocessor


def run_preprocessing():

    print("[*] Starting Improved Preprocessing Pipeline...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_train_path = os.path.join(
        RAW_DIR, "UNSW_NB15_training-set.csv"
    )

    print(" -> Loading official training dataset...")

    train_df = clean_raw_data(pd.read_csv(raw_train_path))

    X = train_df.drop(
        columns=["label", "attack_cat"],
        errors="ignore"
    )

    y = train_df["label"].astype(int)

    print(" -> Splitting raw data into training and validation sets...")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f" -> Training records: {len(X_fit)}")
    print(f" -> Validation records: {len(X_val)}")

    print(" -> Fitting preprocessor only on training data...")

    preprocessor = build_preprocessor(X_fit)

    X_fit_processed = preprocessor.fit_transform(X_fit)
    X_val_processed = preprocessor.transform(X_val)

    feature_names = preprocessor.get_feature_names_out()

    X_fit_df = pd.DataFrame(
        X_fit_processed,
        columns=feature_names
    )

    X_val_df = pd.DataFrame(
        X_val_processed,
        columns=feature_names
    )

    X_fit_df.to_csv(
        os.path.join(PROCESSED_DIR, "X_fit_selection.csv"),
        index=False
    )

    X_val_df.to_csv(
        os.path.join(PROCESSED_DIR, "X_val_selection.csv"),
        index=False
    )

    y_fit.rename("label").to_frame().to_csv(
        os.path.join(PROCESSED_DIR, "y_fit_selection.csv"),
        index=False
    )

    y_val.rename("label").to_frame().to_csv(
        os.path.join(PROCESSED_DIR, "y_val_selection.csv"),
        index=False
    )

    joblib.dump(
        preprocessor,
        os.path.join(PROCESSED_DIR, "preprocessor_selection.joblib")
    )

    print("\n[+] Preprocessing completed successfully.")

    print("\nTraining class distribution:")
    print(y_fit.value_counts().sort_index())

    print("\nValidation class distribution:")
    print(y_val.value_counts().sort_index())

    print("\n[+] Selection datasets and preprocessing pipeline saved.")


if __name__ == "__main__":
    run_preprocessing()