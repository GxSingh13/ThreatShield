
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

from preprocess import clean_raw_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")


def evaluate_model():

    print("[*] Starting Threshold-Aware Final Evaluation...")

    os.makedirs(STATIC_DIR, exist_ok=True)

    print(" -> Loading official testing dataset...")

    test_path = os.path.join(
        RAW_DIR, "UNSW_NB15_testing-set.csv"
    )

    test_df = clean_raw_data(pd.read_csv(test_path))

    X_test = test_df.drop(
        columns=["label", "attack_cat"],
        errors="ignore"
    )

    y_test = test_df["label"].astype(int).values

    print(" -> Loading selected model and preprocessor...")

    model = joblib.load(
        os.path.join(MODELS_DIR, "best_model.joblib")
    )

    preprocessor = joblib.load(
        os.path.join(MODELS_DIR, "preprocessor.joblib")
    )

    with open(
        os.path.join(MODELS_DIR, "best_model_info.json")
    ) as f:
        model_info = json.load(f)

    threshold = float(model_info["decision_threshold"])

    print(f" -> Selected model: {model_info['selected_model']}")
    print(f" -> Validation-selected threshold: {threshold:.2f}")

    print(" -> Transforming testing data...")

    X_test_processed = preprocessor.transform(X_test)

    print(" -> Calculating prediction probabilities...")

    probabilities = model.predict_proba(X_test_processed)[:, 1]

    y_pred = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        pos_label=1,
        zero_division=0
    )

    attack_recall = recall_score(
        y_test, y_pred,
        pos_label=1,
        zero_division=0
    )

    normal_recall = recall_score(
        y_test, y_pred,
        pos_label=0,
        zero_division=0
    )

    attack_f1 = f1_score(
        y_test, y_pred,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test, y_pred
    )

    print("\n========== FINAL TEST RESULTS ==========")

    print(f"Prediction threshold: {threshold:.2f}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Malicious Precision:  {precision:.4f}")
    print(f"Normal Recall:        {normal_recall:.4f}")
    print(f"Attack Recall:        {attack_recall:.4f}")
    print(f"Attack F1-Score:      {attack_f1:.4f}")
    print(f"Macro F1-Score:       {macro_f1:.4f}")
    print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")

    print("\nConfusion Matrix Counts:")

    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\nDetailed Classification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Malicious"],
            zero_division=0
        )
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Malicious (1)"],
        yticklabels=["Normal (0)", "Malicious (1)"]
    )

    plt.title("ThreatShield - Final Confusion Matrix")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()

    cm_path = os.path.join(
        STATIC_DIR, "confusion_matrix.png"
    )

    plt.savefig(cm_path, dpi=300)
    plt.close()

    final_metrics = {
        "Selected Model": model_info["selected_model"],
        "Decision Threshold": threshold,
        "Accuracy": float(accuracy),
        "Malicious Precision": float(precision),
        "Normal Recall": float(normal_recall),
        "Attack Recall": float(attack_recall),
        "Attack F1-Score": float(attack_f1),
        "Macro F1-Score": float(macro_f1),
        "Balanced Accuracy": float(balanced_accuracy),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp)
    }

    metrics_path = os.path.join(
        MODELS_DIR, "final_test_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("\n[+] Final evaluation completed.")
    print(f"[+] Metrics saved to: {metrics_path}")
    print(f"[+] Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    evaluate_model()
