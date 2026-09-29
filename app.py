import os
import time
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CKD Prediction API",
    description="REST API for Chronic Kidney Disease prediction",
    version="1.0.0"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

MODEL_PATH = "models/ckd_pipeline.joblib"
METADATA_PATH = "models/metadata.joblib"

model = None
metadata = None

total_requests = 0
successful_predictions = 0
failed_requests = 0
total_response_time = 0.0


# ============================================================
# INPUT SCHEMA
# ============================================================

class CKDInput(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: str
    pc: str
    pcc: str
    ba: str
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str


# ============================================================
# LOAD MODEL WHEN APPLICATION STARTS
# ============================================================

@app.on_event("startup")
def load_model():

    global model
    global metadata

    print("=" * 60)
    print("STARTING CKD PREDICTION API")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file not found!")
        print("Expected:", MODEL_PATH)
        return

    if not os.path.exists(METADATA_PATH):
        print("ERROR: Metadata file not found!")
        print("Expected:", METADATA_PATH)
        return

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    print("Model loaded successfully!")
    print("Metadata loaded successfully!")
    print("=" * 60)


# ============================================================
# GET /
# ============================================================

@app.get("/")
def root():

    return {
        "service": "CKD Prediction API",
        "version": "1.0.0",
        "status": "running",
        "message": "Welcome to the Chronic Kidney Disease Prediction API"
    }


# ============================================================
# GET /health
# ============================================================

@app.get("/health")
def health():

    if model is not None:

        return {
            "status": "healthy",
            "model_loaded": True
        }

    return {
        "status": "unhealthy",
        "model_loaded": False
    }


# ============================================================
# GET /ready
# ============================================================

@app.get("/ready")
def ready():

    if model is not None:

        return {
            "ready": True,
            "message": "API and model are ready to serve predictions"
        }

    return {
        "ready": False,
        "message": "Model is not loaded"
    }


# ============================================================
# GET /model-info
# ============================================================

@app.get("/model-info")
def model_info():

    if metadata is None:

        raise HTTPException(
            status_code=503,
            detail="Model metadata is not available"
        )

    return {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "algorithm": metadata.get("algorithm"),
        "dataset": metadata.get("dataset"),
        "number_of_features": metadata.get("number_of_features"),
        "training_records": metadata.get("training_records"),
        "testing_records": metadata.get("testing_records"),
        "evaluation_metrics": {
            "accuracy": metadata.get("accuracy"),
            "precision": metadata.get("precision"),
            "recall": metadata.get("recall"),
            "f1_score": metadata.get("f1_score")
        }
    }


# ============================================================
# POST /predict
# ============================================================

@app.post("/predict")
def predict(input_data: CKDInput):

    global total_requests
    global successful_predictions
    global failed_requests
    global total_response_time

    start_time = time.time()

    total_requests += 1

    try:

        if model is None:

            failed_requests += 1

            raise HTTPException(
                status_code=503,
                detail="Model is not loaded"
            )

        # Convert input into dictionary
        input_dictionary = input_data.model_dump()

        # Convert dictionary into DataFrame
        input_df = pd.DataFrame([input_dictionary])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Get probability if supported
        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            probability = float(max(probabilities))

        # Convert prediction into readable output
        if int(prediction) == 1:

            result = "CKD"

        else:

            result = "Not CKD"

        successful_predictions += 1

        response_time = time.time() - start_time
        total_response_time += response_time

        return {
            "prediction": result,
            "prediction_code": int(prediction),
            "model_probability": probability,
            "message": "Prediction generated by the machine learning model. This is not a medical diagnosis."
        }

    except HTTPException:

        raise

    except Exception as e:

        failed_requests += 1

        response_time = time.time() - start_time
        total_response_time += response_time

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# GET /metrics
# ============================================================

@app.get("/metrics")
def metrics():

    if total_requests > 0:

        average_response_time = (
            total_response_time / total_requests
        )

    else:

        average_response_time = 0.0

    return {
        "total_requests": total_requests,
        "successful_predictions": successful_predictions,
        "failed_requests": failed_requests,
        "average_response_time_seconds": round(
            average_response_time,
            4
        )
    }


# ============================================================
# APPLICATION END
# ============================================================