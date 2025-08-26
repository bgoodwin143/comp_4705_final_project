import os
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import wandb
import boto3
from botocore.exceptions import ClientError
import logging
from contextlib import asynccontextmanager
import time  # NEW: Import the time module

# --- Application Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App State ---
app_state = {"pipeline": None, "dynamodb_table": None}


# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block runs on startup
    logger.info("--- Application Startup ---")

    # Load Model
    logger.info("Downloading model from W&B...")
    try:
        run = wandb.init(
            project="Toxic-Comment-Classification-Final", job_type="inference"
        )
        artifact = run.use_artifact(
            "bensharn-university-of-denver/Toxic-Comment-Classification-Final/toxic-comment-pipeline:production",
            type="model",
        )
        artifact_dir = artifact.download()
        pipeline_path = os.path.join(artifact_dir, "toxic-comment-pipeline.pkl")
        app_state["pipeline"] = joblib.load(pipeline_path)
        logger.info("Production model pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from W&B. Error: {e}")
    finally:
        if wandb.run:
            wandb.finish()

    # Connect to DynamoDB
    DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "prediction_logs")
    try:
        dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        table.load()
        app_state["dynamodb_table"] = table
        logger.info(f"Successfully connected to DynamoDB table: {DYNAMODB_TABLE_NAME}")
    except ClientError as e:
        logger.error(f"Failed to connect to DynamoDB: {e.response['Error']['Message']}")

    yield

    # This block runs on shutdown
    logger.info("--- Application Shutdown ---")


# Initialize FastAPI app
app = FastAPI(
    title="Toxic Comment Classification API", version="1.0.0", lifespan=lifespan
)


# --- Dependency Functions ---
def get_pipeline():
    if app_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    return app_state["pipeline"]


def get_db_table():
    if app_state["dynamodb_table"] is None:
        raise HTTPException(status_code=503, detail="Database is not available.")
    return app_state["dynamodb_table"]


# --- API Data Models ---
class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction_id: str
    classification: str


# Pydantic model for the feedback request body
class FeedbackRequest(BaseModel):
    prediction_id: str
    feedback: str  # "correct" or "incorrect"


# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check(pipeline=Depends(get_pipeline), table=Depends(get_db_table)):
    return {"status": "ok", "model_ready": True, "database_ready": True}


@app.post("/predict", response_model=PredictionResponse, summary="Classify a comment")
def predict(
    request: PredictionRequest,
    pipeline=Depends(get_pipeline),
    table=Depends(get_db_table),
):
    start_time = time.time()  # NEW: Start timer

    prediction_id = str(uuid.uuid4())
    prediction_array = pipeline.predict([request.text])
    classification_result = "toxic" if prediction_array[0] == 1 else "not_toxic"

    end_time = time.time()  # NEW: End timer
    latency_ms = round((end_time - start_time) * 1000, 2)  # NEW: Calculate latency

    log_item = {
        "prediction_id": prediction_id,
        "text_input": request.text,
        "classification": classification_result,
        "prediction_latency_ms": latency_ms,  # NEW: Add latency to log
        "model_artifact_used": "bensharn-university-of-denver/Toxic-Comment-Classification-Final/toxic-comment-pipeline:production",
        "timestamp": pd.Timestamp.now().isoformat(),
        "user_feedback": "N/A",
    }

    try:
        table.put_item(Item=log_item)
        logger.info(f"Logged prediction {prediction_id} to DynamoDB.")
    except ClientError as e:
        logger.error(f"DynamoDB put_item failed: {e.response['Error']['Message']}")

    return PredictionResponse(
        prediction_id=prediction_id, classification=classification_result
    )


# Endpoint to receive user feedback
@app.post("/feedback", summary="Submit feedback for a prediction")
def submit_feedback(
    request: FeedbackRequest,
    table=Depends(get_db_table),
):
    try:
        table.update_item(
            Key={"prediction_id": request.prediction_id},
            UpdateExpression="SET user_feedback = :val",
            ExpressionAttributeValues={":val": request.feedback},
        )
        logger.info(
            f"Updated feedback for {request.prediction_id} to '{request.feedback}'"
        )
        return {"status": "success", "message": "Feedback submitted."}
    except ClientError as e:
        logger.error(f"DynamoDB update_item failed: {e.response['Error']['Message']}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback.")
