from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# File Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "blood_donation.csv"
MODEL_PATH = BASE_DIR / "model_bundle.pkl"


# -----------------------------
# Load Dataset
# -----------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Remove empty rows and duplicate records
df = df.dropna(how="all").drop_duplicates()

if df.empty:
    raise ValueError("The dataset is empty.")

if df.shape[1] < 2:
    raise ValueError(
        "Dataset must contain at least one feature column and one target column."
    )


# -----------------------------
# Target Column
# -----------------------------
# Change this value if your CSV uses a different target column.
TARGET_COLUMN = df.columns[-1]

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found in the dataset."
    )


# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

if y.nunique() < 2:
    raise ValueError(
        "The target column must contain at least two classes."
    )


# -----------------------------
# Validate Feature Columns
# -----------------------------
non_numeric = [
    column
    for column in X.columns
    if not pd.api.types.is_numeric_dtype(X[column])
]

if non_numeric:
    raise ValueError(
        "Non-numeric feature columns found: "
        + ", ".join(non_numeric)
        + ". Please encode these columns before training."
    )

numeric_features = list(X.columns)


# -----------------------------
# Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
)


# -----------------------------
# Machine Learning Model
# -----------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
        ),
    ]
)


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------
predictions = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 50)
print("BLOOD DONOR PREDICTION - MODEL RESULTS")
print("=" * 50)

print(f"Target Column : {TARGET_COLUMN}")
print(f"Training Data : {len(X_train)} records")
print(f"Testing Data  : {len(X_test)} records")
print(f"Accuracy      : {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# -----------------------------
# Save Model
# -----------------------------
model_bundle = {
    "model": model,
    "feature_names": numeric_features,
    "target_column": TARGET_COLUMN,
}

joblib.dump(model_bundle, MODEL_PATH)

print("\n" + "=" * 50)
print(f"Model saved successfully: {MODEL_PATH}")
print("=" * 50)
