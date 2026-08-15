from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("dataset/blood_donation.csv")
MODEL_PATH = Path("model_bundle.pkl")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Add your CSV file as dataset/blood_donation.csv"
    )

df = pd.read_csv(DATA_PATH)
df = df.dropna(how="all").drop_duplicates()

if df.shape[1] < 2:
    raise ValueError("Dataset must contain feature columns and one target column.")

# By default, the last column is treated as the target.
# Change TARGET_COLUMN if your dataset uses a different target column.
TARGET_COLUMN = df.columns[-1]

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# This starter project expects numeric input features.
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
if non_numeric:
    raise ValueError(
        "Non-numeric feature columns found: "
        + ", ".join(non_numeric)
        + ". Encode them before training or update the preprocessing pipeline."
    )

numeric_features = list(X.columns)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
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

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() > 1 else None,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Target column: {TARGET_COLUMN}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))

bundle = {
    "model": model,
    "feature_names": numeric_features,
    "target_column": TARGET_COLUMN,
}

joblib.dump(bundle, MODEL_PATH)
print(f"Saved trained model to: {MODEL_PATH}")
