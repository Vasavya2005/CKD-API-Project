import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load(
    "models/ckd_pipeline.joblib"
)

print("Saved model loaded successfully!")

# Load dataset
df = pd.read_csv(
    "data/kidney_disease.csv"
)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Remove ID
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# Remove target
X = df.drop(
    columns=["classification"]
)

# Handle missing value representations
X = X.replace(
    ["?", "\t?", " ?", "", "nan", "NaN", "NA", "N/A"],
    np.nan
)

# Clean categorical values
for column in X.select_dtypes(
    include="object"
).columns:

    X[column] = X[column].apply(
        lambda x: x.strip().lower()
        if isinstance(x, str)
        else x
    )

# Convert numerical columns
numeric_columns = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc"
]

for column in numeric_columns:

    if column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

# Take one patient record
sample = X.iloc[[0]]

# Predict
prediction = model.predict(sample)

print("\nPrediction value:", prediction[0])

if prediction[0] == 1:
    print("Prediction: CKD")
else:
    print("Prediction: Not CKD")