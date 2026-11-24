# ThreatShield
## Machine Learning-Based Network Intrusion Detection System

ThreatShield is an academic network intrusion detection prototype built with the UNSW-NB15 dataset. It uses a trained machine-learning classifier to classify uploaded network-traffic records as **Normal** or **Malicious**.

The final web interface is a **Flask application**. The older Streamlit interface and unrelated live-feed prototype files have been excluded from this distribution.

## Main components

- `web_app.py` — Flask web application for CSV upload, prediction and report download.
- `templates/` — HTML pages for upload and prediction results.
- `static/` — CSS and project visual assets.
- `models/best_model.joblib` — trained Random Forest classifier.
- `models/preprocessor.joblib` — fitted feature preprocessing pipeline.
- `models/best_model_info.json` — selected model information and decision threshold.
- `models/final_test_metrics.json` — final evaluation metrics.
- `src/preprocess.py` — data-cleaning and preprocessing pipeline.
- `src/train.py` — model comparison and training script.
- `src/evaluate.py` — final model evaluation script.

## Model summary

- Class `0`: Normal
- Class `1`: Malicious
- Classifier: Random Forest
- Number of trees: 100
- Maximum depth: 20
- Minimum samples per leaf: 2
- `max_features`: `sqrt`
- Decision threshold: `0.64`

The saved model was trained using the official UNSW-NB15 training dataset. The reported final test accuracy is approximately 90.50%, with Attack Recall of approximately 96.63%.

## Requirements

- Python 3.13 (or a compatible Python version)
- Dependencies listed in `requirements.txt`

The saved scikit-learn model was serialized with scikit-learn 1.9.1. Use the specified version to avoid model-persistence compatibility warnings.

## Installation and run

Open PowerShell in the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python web_app.py
```

Then open the local address printed by Flask (normally `http://127.0.0.1:5000`).

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Using the web application

1. Open the Flask homepage.
2. Select a CSV file containing the required network-traffic features.
3. Submit the file for analysis.
4. View the Normal/Malicious prediction summary and record-level preview.
5. Download the generated CSV report.

The uploaded CSV must contain the 42 original model input features expected by the saved preprocessor. Extra columns may be present; the application selects the required model features. The application accepts CSV files up to 32 MB.

This is batch CSV inference. It does not perform live packet capture.

## Reproducing training

The raw UNSW-NB15 CSV files are not bundled with this project archive. To rerun preprocessing, training or evaluation, obtain the official UNSW-NB15 training and testing CSV files and place them in:

```text
data/
└── raw/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

Then run the relevant scripts from the project root. Training and evaluation are not required to run the already-trained Flask application.

## Important note

The application is intended for academic demonstration. Before any public deployment, disable Flask debug mode and add suitable authentication, access controls and production hosting configuration.
