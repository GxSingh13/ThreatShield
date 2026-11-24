
import os
import json
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

from preprocess import clean_raw_data, build_preprocessor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MIN_ATTACK_RECALL = 0.95


def create_candidates():

    return {
        "Logistic Regression - Standard": lambda: LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Logistic Regression - Balanced": lambda: LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree - Standard": lambda: DecisionTreeClassifier(
            max_depth=12,
            random_state=42
        ),

        "Decision Tree - Balanced": lambda: DecisionTreeClassifier(
            max_depth=12,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest - Standard": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        ),

        "Random Forest - Balanced": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "Random Forest - Depth 20 Leaf 2": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),

        "Random Forest - Depth 15 Leaf 5": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
            random_state=42
        )
    }


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    normal_recall = tn / (tn + fp) if (tn + fp) else 0.0
    attack_recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Normal Recall": float(normal_recall),
        "Attack Recall": float(attack_recall),
        "Balanced Accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "Macro F1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "Attack F1": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            )
        )
    }


def choose_threshold(y_true, probabilities):

    results = []

    thresholds = np.round(
        np.arange(0.10, 0.951, 0.01),
        2
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        if metrics["Attack Recall"] >= MIN_ATTACK_RECALL:
            results.append(metrics)

    if results:

        selected = max(
            results,
            key=lambda item: (
                item["Normal Recall"],
                item["Balanced Accuracy"],
                item["Macro F1"],
                item["Accuracy"]
            )
        )

        return selected, True

    all_results = [
        calculate_metrics(y_true, probabilities, threshold)
        for threshold in thresholds
    ]

    selected = max(
        all_results,
        key=lambda item: (
            item["Balanced Accuracy"],
            item["Macro F1"],
            item["Attack Recall"]
        )
    )

    return selected, False


def train_and_select_model():

    print("[*] Starting Advanced Model Optimization...")

    os.makedirs(MODELS_DIR, exist_ok=True)

    X_fit = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_fit_selection.csv")
    )

    X_val = pd.read_csv(
        os.path.join(PROCESSED_DIR, "X_val_selection.csv")
    )

    y_fit = pd.read_csv(
        os.path.join(PROCESSED_DIR, "y_fit_selection.csv")
    )["label"].values

    y_val = pd.read_csv(
        os.path.join(PROCESSED_DIR, "y_val_selection.csv")
    )["label"].values

    candidates = create_candidates()

    results = {}

    best_model_name = None
    best_threshold = None
    best_metrics = None
    best_model_factory = None
    best_target_met = False

    print("\n[*] Comparing eight model configurations...")

    print(
        f"[*] Target: maximize Normal Recall while maintaining "
        f"Attack Recall >= {MIN_ATTACK_RECALL:.0%}"
    )

    for name, model_factory in candidates.items():

        print(f"\n -> Training {name}...")

        clf = model_factory()

        clf.fit(X_fit, y_fit)

        probabilities = clf.predict_proba(X_val)[:, 1]

        selected_metrics, target_met = choose_threshold(
            y_val,
            probabilities
        )

        results[name] = {
            **selected_metrics,
            "Attack Recall Target Met": bool(target_met)
        }

        print(
            f"    Selected threshold: "
            f"{selected_metrics['Threshold']:.2f}"
        )

        print(
            f"    Accuracy: "
            f"{selected_metrics['Accuracy']:.4f}"
        )

        print(
            f"    Normal Recall: "
            f"{selected_metrics['Normal Recall']:.4f}"
        )

        print(
            f"    Attack Recall: "
            f"{selected_metrics['Attack Recall']:.4f}"
        )

        print(
            f"    Balanced Accuracy: "
            f"{selected_metrics['Balanced Accuracy']:.4f}"
        )

        print(
            f"    Macro F1: "
            f"{selected_metrics['Macro F1']:.4f}"
        )

        if best_metrics is None:

            is_better = True

        else:

            is_better = (
                target_met,
                selected_metrics["Balanced Accuracy"],
                selected_metrics["Macro F1"],
                selected_metrics["Normal Recall"],
                selected_metrics["Attack Recall"]
            ) > (
                best_target_met,
                best_metrics["Balanced Accuracy"],
                best_metrics["Macro F1"],
                best_metrics["Normal Recall"],
                best_metrics["Attack Recall"]
            )

        if is_better:

            best_model_name = name
            best_threshold = selected_metrics["Threshold"]
            best_metrics = selected_metrics
            best_model_factory = model_factory
            best_target_met = target_met

        del clf, probabilities
        gc.collect()

    print("\n========== VALIDATION CHAMPION ==========")

    print(f"Selected model: {best_model_name}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Normal Recall: {best_metrics['Normal Recall']:.4f}")
    print(f"Attack Recall: {best_metrics['Attack Recall']:.4f}")
    print(
        f"Balanced Accuracy: "
        f"{best_metrics['Balanced Accuracy']:.4f}"
    )
    print(f"Macro F1: {best_metrics['Macro F1']:.4f}")

    print(
        f"95% Attack Recall Target Met: {best_target_met}"
    )

    results_path = os.path.join(
        MODELS_DIR,
        "metrics_summary.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n[*] Loading complete official training dataset...")

    raw_train_path = os.path.join(
        RAW_DIR,
        "UNSW_NB15_training-set.csv"
    )

    full_train_df = clean_raw_data(
        pd.read_csv(raw_train_path)
    )

    X_full = full_train_df.drop(
        columns=["label", "attack_cat"],
        errors="ignore"
    )

    y_full = full_train_df["label"].astype(int).values

    print(
        "[*] Fitting final preprocessor on complete training data..."
    )

    final_preprocessor = build_preprocessor(X_full)

    X_full_processed = final_preprocessor.fit_transform(X_full)

    print("[*] Training final selected model...")

    final_model = best_model_factory()

    final_model.fit(X_full_processed, y_full)

    joblib.dump(
        final_preprocessor,
        os.path.join(MODELS_DIR, "preprocessor.joblib")
    )

    joblib.dump(
        final_model,
        os.path.join(MODELS_DIR, "best_model.joblib")
    )

    model_info = {
        "selected_model": best_model_name,
        "selection_metric": "Validation Balanced Accuracy",
        "decision_threshold": float(best_threshold),
        "minimum_attack_recall_target": MIN_ATTACK_RECALL,
        "validation_metrics": best_metrics,
        "validation_attack_recall_target_met": bool(best_target_met),
        "final_training_records": int(len(X_full))
    }

    with open(
        os.path.join(MODELS_DIR, "best_model_info.json"),
        "w"
    ) as f:
        json.dump(model_info, f, indent=4)

    print("\n[+] Final model and preprocessing pipeline saved.")

    print(f"[+] Selected model: {best_model_name}")
    print(f"[+] Prediction threshold: {best_threshold:.2f}")

    print("[+] Training and optimization completed.")


if __name__ == "__main__":
    train_and_select_model()
