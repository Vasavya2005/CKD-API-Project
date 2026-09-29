import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("=" * 60)
print("TASK 1: DATASET AND MODEL PREPARATION")
print("=" * 60)

print("\n1. Loading dataset...")

df = pd.read_csv("data/kidney_disease.csv")

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ==========================================================
# 2. CLEAN COLUMN NAMES
# ==========================================================

print("\n2. Cleaning column names...")

df.columns = df.columns.str.strip().str.lower()

print("Columns:")
print(df.columns.tolist())


# ==========================================================
# 3. REMOVE ID COLUMN
# ==========================================================

print("\n3. Removing unnecessary columns...")

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

print("Dataset shape after removing ID:", df.shape)


# ==========================================================
# 4. HANDLE MISSING VALUES
# ==========================================================

print("\n4. Handling missing values...")

missing_values = [
    "?",
    "\t?",
    " ?",
    "",
    "nan",
    "NaN",
    "NA",
    "N/A"
]

df = df.replace(missing_values, np.nan)

# Remove extra spaces and convert text to lowercase
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].apply(
        lambda x: x.strip().lower()
        if isinstance(x, str)
        else x
    )

print("Missing values handled.")


# ==========================================================
# 5. PREPARE TARGET COLUMN
# ==========================================================

print("\n5. Preparing target column...")

target_column = "classification"

if target_column not in df.columns:
    raise ValueError(
        "The 'classification' column was not found. "
        "Please check your dataset."
    )

# Clean target values
df[target_column] = df[target_column].replace({
    "ckd": "ckd",
    "ckd\t": "ckd",
    "notckd": "notckd",
    "not ckd": "notckd",
    "notckd\t": "notckd"
})

# Remove rows where target is missing
df = df.dropna(subset=[target_column])

# Keep only valid target classes
df = df[
    df[target_column].isin(["ckd", "notckd"])
]

# Convert target to numbers
# CKD = 1
# Not CKD = 0

df[target_column] = df[target_column].map({
    "ckd": 1,
    "notckd": 0
})

print("Target distribution:")
print(df[target_column].value_counts())


# ==========================================================
# 6. SEPARATE INPUT FEATURES AND TARGET
# ==========================================================

print("\n6. Separating features and target...")

X = df.drop(columns=[target_column])
y = df[target_column]

print("Number of records:", len(X))
print("Number of input features:", X.shape[1])


# ==========================================================
# 7. CONVERT NUMERICAL FEATURES
# ==========================================================

print("\n7. Converting numerical features...")

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


# ==========================================================
# 8. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================================

print("\n8. Identifying feature types...")

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================================
# 9. CREATE NUMERICAL PREPROCESSING
# ==========================================================

print("\n9. Creating numerical preprocessing...")

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ==========================================================
# 10. CREATE CATEGORICAL PREPROCESSING
# ==========================================================

print("Creating categorical preprocessing...")

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ==========================================================
# 11. COMBINE PREPROCESSING
# ==========================================================

print("Combining preprocessing pipelines...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================================
# 12. CREATE RANDOM FOREST MODEL
# ==========================================================

print("\n10. Creating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ==========================================================
# 13. CREATE COMPLETE ML PIPELINE
# ==========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

print("ML pipeline created.")


# ==========================================================
# 14. SPLIT DATASET
# ==========================================================

print("\n11. Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================================
# 15. TRAIN MODEL
# ==========================================================

print("\n12. Training Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ==========================================================
# 16. MAKE PREDICTIONS
# ==========================================================

print("\n13. Making predictions...")

y_pred = pipeline.predict(X_test)

print("Predictions completed.")


# ==========================================================
# 17. CALCULATE EVALUATION METRICS
# ==========================================================

print("\n14. Evaluating model...")

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ==========================================================
# 18. DISPLAY RESULTS
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not CKD", "CKD"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================================
# 19. SAVE MODEL
# ==========================================================

print("\n15. Saving trained model...")

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/ckd_pipeline.joblib"

joblib.dump(
    pipeline,
    model_path
)

print(
    "Model saved successfully:",
    model_path
)


# ==========================================================
# 20. SAVE MODEL METADATA
# ==========================================================

print("\n16. Saving model metadata...")

metadata = {
    "model_name": "CKD Prediction Model",
    "model_version": "1.0.0",
    "algorithm": "Random Forest Classifier",
    "dataset": "Chronic Kidney Disease Dataset",
    "number_of_features": int(X.shape[1]),
    "training_records": int(X_train.shape[0]),
    "testing_records": int(X_test.shape[0]),
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4)
}

joblib.dump(
    metadata,
    "models/metadata.joblib"
)

print("Metadata saved successfully.")


# ==========================================================
# 21. FINAL RESULT
# ==========================================================

print("\n")
print("=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Dataset loaded")
print("Data cleaned")
print("Missing values handled")
print("Data preprocessing completed")
print("Train-test split completed")
print("Random Forest model trained")
print("Model evaluated")
print("Model saved using Joblib")
print("Model metadata saved")

print("\nFiles created:")
print("models/ckd_pipeline.joblib")
print("models/metadata.joblib")