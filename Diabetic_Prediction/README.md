# Diabetic Prediction Flask App

This project trains a **Two-Class classification** model using the `diabetes.csv` file and serves predictions through a Flask web interface.

The training algorithm used is:
- Two-Class Boosted Decision Tree (implemented with `sklearn.ensemble.GradientBoostingClassifier`)

The label predicted by the model is:
- `Diabetic`

## Project Files

- `app.py`: Flask server, model training, validation, and prediction logic.
- `templates/index.html`: Web UI for user inputs and prediction output.
- `static/style.css`: Styling for the UI.
- `diabetes.csv`: Dataset.

## Prerequisites

- Python 3.9+ recommended

## Install Dependencies

Run the following command in the project folder:

```bash
pip install -r requirements.txt
```

## Run the Application

From the project directory:

```bash
python app.py
```

Flask will start a local server (usually `http://127.0.0.1:5000`).

## How It Works

1. On startup, `app.py` loads `diabetes.csv`.
2. It builds a Two-Class Boosted Decision Tree model.
3. It performs validation with:
- Holdout split metrics: Accuracy, Precision, Recall, F1, ROC AUC.
- 5-fold cross-validation metrics: Accuracy and ROC AUC mean.
4. The web page shows validation metrics.
5. You enter feature values and submit the form to get:
- Predicted class (`Diabetic` or `Non-Diabetic`)
- Prediction probability for diabetic class.

## Input Fields in UI

The UI includes numeric fields based on dataset features except:
- `Diabetic` (label)
- `PatientID` (ignored as an identifier)

## Notes

- If model training fails, the UI displays the training error.
- Missing values are handled with median imputation during training.
