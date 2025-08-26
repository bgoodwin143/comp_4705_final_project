# api/main.py
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

# --- Application Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App State ---
# Use a dictionary to hold app state instead of global variables
app_state = {"pipeline": None, "dynamodb_table": None}


# --- Lifespan Events (The new way to do startup/shutdown) ---
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
        # --- START OF TEMPORARY CHANGE ---
        logger.info(f"DEBUG: Artifact downloaded to ==> {artifact_dir}")
        logger.info(f"DEBUG: Contents of that directory ==> {os.listdir(artifact_dir)}")
        # --- END OF TEMPORARY CHANGE ---
        pipeline_path = os.path.join(artifact_dir, "toxic_comment_pipeline.pkl")
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

    yield  # The application runs here

    # This block runs on shutdown
    logger.info("--- Application Shutdown ---")
    app_state["pipeline"] = None
    app_state["dynamodb_table"] = None


# Initialize FastAPI app with the lifespan manager
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
    prediction_id = str(uuid.uuid4())
    prediction_array = pipeline.predict([request.text])
    classification_result = "toxic" if prediction_array[0] == 1 else "not_toxic"

    log_item = {
        "prediction_id": prediction_id,
        "text_input": request.text,
        "classification": classification_result,
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
