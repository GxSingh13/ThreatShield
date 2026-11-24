
from pathlib import Path
import json
import uuid

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    abort,
    url_for
)
from werkzeug.utils import secure_filename


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
MODEL_INFO_PATH = BASE_DIR / "models" / "best_model_info.json"

OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

print("Loading ThreatShield model...")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(MODEL_INFO_PATH, "r", encoding="utf-8") as file:
    model_info = json.load(file)

DECISION_THRESHOLD = float(model_info["decision_threshold"])

FEATURE_NAMES = list(preprocessor.feature_names_in_)

CATEGORICAL_COLUMNS = ["proto", "service", "state"]

MALICIOUS_CLASS_INDEX = list(model.classes_).index(1)

print("Model loaded successfully!")
print("Model:", model_info.get("selected_model"))
print("Number of features:", len(FEATURE_NAMES))
print("Decision threshold:", DECISION_THRESHOLD)


# --------------------------------------------------
# 3. HOMEPAGE AND PREDICTION
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        uploaded_file = request.files.get("traffic_file")

        if uploaded_file is None or uploaded_file.filename == "":
            return render_template(
                "index.html",
                error_message="Please select a CSV file."
            )

        if not uploaded_file.filename.lower().endswith(".csv"):
            return render_template(
                "index.html",
                error_message="Please upload a CSV file."
            )

        try:

            # Read uploaded CSV.
            traffic_df = pd.read_csv(
                uploaded_file,
                low_memory=False
            )

            if traffic_df.empty:
                return render_template(
                    "index.html",
                    error_message="The uploaded CSV contains no records."
                )

            # Check required features.
            missing_columns = [
                col for col in FEATURE_NAMES
                if col not in traffic_df.columns
            ]

            if missing_columns:
                return render_template(
                    "index.html",
                    error_message=(
                        "Missing required features: "
                        + ", ".join(missing_columns)
                    )
                )

            # Select the exact 42 model features.
            X = traffic_df[FEATURE_NAMES].copy()

            # Clean categorical values in the same way as training.
            for col in CATEGORICAL_COLUMNS:
                X[col] = (
                    X[col]
                    .replace("-", "unknown")
                    .fillna("unknown")
                    .astype(str)
                )

            # --------------------------------------------------
            # 4. MACHINE LEARNING PREDICTIONS
            # --------------------------------------------------

            X_processed = preprocessor.transform(X)

            malicious_scores = model.predict_proba(
                X_processed
            )[:, MALICIOUS_CLASS_INDEX]

            predictions = (
                malicious_scores >= DECISION_THRESHOLD
            ).astype(int)

            labels = np.where(
                predictions == 1,
                "Malicious",
                "Normal"
            )

            # --------------------------------------------------
            # 5. CALCULATE STATISTICS
            # --------------------------------------------------

            total_records = len(predictions)

            normal_count = int(np.sum(predictions == 0))
            malicious_count = int(np.sum(predictions == 1))

            normal_pct = round(
                normal_count / total_records * 100, 1
            )

            malicious_pct = round(
                malicious_count / total_records * 100, 1
            )

            # --------------------------------------------------
            # 6. CREATE DOWNLOADABLE PREDICTION REPORT
            # --------------------------------------------------

            report_df = traffic_df.copy()

            report_df["ThreatShield_Prediction"] = labels

            report_df["Malicious_Probability_Percent"] = np.round(
                malicious_scores * 100, 2
            )

            report_id = uuid.uuid4().hex

            report_path = (
                OUTPUT_DIR / f"predictions_{report_id}.csv"
            )

            report_df.to_csv(
                report_path,
                index=False
            )

            # --------------------------------------------------
            # 7. PREPARE FIRST 100 RESULTS FOR THE WEBPAGE
            # --------------------------------------------------

            preview_results = []

            for index in range(min(100, total_records)):

                preview_results.append({
                    "record_number": index + 1,
                    "prediction": labels[index],
                    "malicious_probability": round(
                        float(malicious_scores[index]) * 100, 2
                    )
                })

            return render_template(
                "results.html",
                filename=secure_filename(uploaded_file.filename),
                total_records=total_records,
                normal_count=normal_count,
                malicious_count=malicious_count,
                normal_pct=normal_pct,
                malicious_pct=malicious_pct,
                preview_results=preview_results,
                shown_records=len(preview_results),
                threshold=DECISION_THRESHOLD,
                download_url=url_for(
                    "download_report",
                    report_id=report_id
                )
            )

        except pd.errors.EmptyDataError:
            error_message = "The uploaded CSV file is empty."

        except pd.errors.ParserError:
            error_message = "Unable to read the CSV file format."

        except Exception:
            app.logger.exception("Prediction error occurred.")

            error_message = (
                "Prediction failed. Please check your CSV features "
                "and data values."
            )

        return render_template(
            "index.html",
            error_message=error_message
        )

    return render_template("index.html")


# --------------------------------------------------
# 8. DOWNLOAD PREDICTION REPORT
# --------------------------------------------------

@app.route("/download/<report_id>")
def download_report(report_id):

    # Accept only a generated 32-character hexadecimal ID.
    if (
        len(report_id) != 32
        or any(c not in "0123456789abcdef" for c in report_id.lower())
    ):
        abort(404)

    report_path = (
        OUTPUT_DIR / f"predictions_{report_id}.csv"
    )

    if not report_path.is_file():
        abort(404)

    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"ThreatShield_predictions_{report_id[:8]}.csv",
        mimetype="text/csv"
    )


# --------------------------------------------------
# 9. FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error_message="File too large. Maximum size is 32 MB."
    ), 413


# --------------------------------------------------
# 10. START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
