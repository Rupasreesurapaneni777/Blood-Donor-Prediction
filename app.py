from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# -----------------------------
# File Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_bundle.pkl"


# -----------------------------
# Flask Application
# -----------------------------
app = Flask(__name__)


# -----------------------------
# Load Trained Model
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Please run 'python train_model.py' first."
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_names = bundle["feature_names"]
target_column = bundle.get("target_column", "Donated_Blood")


# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Collect input values from the form
            input_data = {}

            for feature in feature_names:
                value = request.form.get(feature)

                if value is None or value.strip() == "":
                    raise ValueError(
                        f"Please enter a value for '{feature}'."
                    )

                input_data[feature] = float(value)

            # Convert input into DataFrame
            input_df = pd.DataFrame(
                [input_data],
                columns=feature_names
            )

            # Make prediction
            result = model.predict(input_df)[0]

            # Convert prediction to user-friendly message
            if str(result).lower() in {
                "1",
                "1.0",
                "true",
                "yes"
            }:
                prediction = "Likely to donate blood again"

            elif str(result).lower() in {
                "0",
                "0.0",
                "false",
                "no"
            }:
                prediction = "Not likely to donate blood again"

            else:
                prediction = f"Predicted class: {result}"

        except ValueError as exc:
            error = str(exc)

        except Exception as exc:
            error = (
                "An error occurred while making the prediction. "
                f"Details: {exc}"
            )

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=prediction,
        error=error,
        target_column=target_column,
    )


# -----------------------------
# Run Flask Application
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
