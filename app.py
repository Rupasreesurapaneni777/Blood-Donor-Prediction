from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

MODEL_PATH = Path("model_bundle.pkl")

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model_bundle.pkl not found. Run `python train_model.py` first."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = {}
            for feature in feature_names:
                values[feature] = float(request.form[feature])

            input_df = pd.DataFrame([values])
            result = model.predict(input_df)[0]

            # Works for common 0/1 targets. If your labels are different,
            # the raw predicted label will still be shown.
            if str(result) in {"1", "1.0", "True", "true", "Yes", "yes"}:
                prediction = "Likely to donate blood again"
            elif str(result) in {"0", "0.0", "False", "false", "No", "no"}:
                prediction = "Not likely to donate blood again"
            else:
                prediction = f"Predicted class: {result}"

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
