from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes.csv"
LABEL_COLUMN = "Diabetic"
IGNORE_COLUMNS = {"PatientID"}

MODEL = None
FEATURE_COLUMNS = []
METRICS = {}
MODEL_ERROR = None


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load data and return cleaned feature matrix, label vector, and feature names."""
    df = pd.read_csv(DATA_PATH)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected label column '{LABEL_COLUMN}' in {DATA_PATH.name}")

    candidate_features = [
        col for col in df.columns if col != LABEL_COLUMN and col not in IGNORE_COLUMNS
    ]

    if not candidate_features:
        raise ValueError("No feature columns available after excluding label and ignored columns")

    X = df[candidate_features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    # Replace missing numeric values with median per column to keep training robust.
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.mode().iloc[0]).astype(int)

    return X, y, candidate_features


def train_and_validate() -> tuple[GradientBoostingClassifier, dict, list[str]]:
    """Train a Two-Class Boosted Decision Tree model and collect validation metrics."""
    X, y, feature_columns = load_dataset()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_prob = model.predict_proba(X_valid)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {
        "validation_accuracy": accuracy_score(y_valid, y_pred),
        "validation_precision": precision_score(y_valid, y_pred, zero_division=0),
        "validation_recall": recall_score(y_valid, y_pred, zero_division=0),
        "validation_f1": f1_score(y_valid, y_pred, zero_division=0),
        "validation_roc_auc": roc_auc_score(y_valid, y_prob),
        "cv_accuracy_mean": cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean(),
        "cv_roc_auc_mean": cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean(),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
    }

    return model, metrics, feature_columns


def initialize_model() -> None:
    global MODEL, FEATURE_COLUMNS, METRICS, MODEL_ERROR
    try:
        MODEL, METRICS, FEATURE_COLUMNS = train_and_validate()
        MODEL_ERROR = None
    except Exception as exc:  # pragma: no cover - fallback path for runtime issues
        MODEL = None
        FEATURE_COLUMNS = []
        METRICS = {}
        MODEL_ERROR = str(exc)


initialize_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    form_error = None

    if request.method == "POST":
        if MODEL is None:
            form_error = f"Model is unavailable: {MODEL_ERROR}"
        else:
            try:
                values = []
                for feature in FEATURE_COLUMNS:
                    raw = request.form.get(feature, "").strip()
                    if raw == "":
                        raise ValueError(f"{feature} is required")
                    values.append(float(raw))

                input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
                pred = int(MODEL.predict(input_df)[0])
                prob = float(MODEL.predict_proba(input_df)[0][1])

                prediction = "Diabetic" if pred == 1 else "Non-Diabetic"
                probability = prob
            except ValueError as exc:
                form_error = str(exc)
            except Exception as exc:  # pragma: no cover - fallback path for runtime issues
                form_error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        features=FEATURE_COLUMNS,
        metrics=METRICS,
        prediction=prediction,
        probability=probability,
        form_error=form_error,
        model_error=MODEL_ERROR,
    )


if __name__ == "__main__":
    app.run(debug=True)
