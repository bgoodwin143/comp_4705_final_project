import os
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import wandb
import boto3
from botocore.exceptions import ClientError
import logging

# --- Application Setup ---
# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI application
app = FastAPI(title="Toxic Comment Classification API", version="1.0.0")

# --- Model Loading (at Startup) ---
# This block of code runs only once when the API server starts.
# It ensures the model is loaded and ready before accepting requests.

# Define the W&B artifact path. Using ':production' is best practice.
MODEL_ARTIFACT_PATH = "bensharn-university-of-denver/Toxic-Comment-Classification-Final/toxic-comment-pipeline:production"
pipeline = None

@app.on_event("startup")
def load_production_model():
    """
    This function is called when the application starts.
    It downloads the production-tagged model artifact from W&B,
    and loads the scikit-learn pipeline into memory.
    """
    global pipeline
    logger.info(f"Attempting to download model artifact: {MODEL_ARTIFACT_PATH}")
    try:
        # Initialize a W&B run to interact with artifacts
        run = wandb.init(project="Toxic-Comment-Classification-Final", job_type="inference")
        # Use the artifact from the registry
        artifact = run.use_artifact(MODEL_ARTIFACT_PATH, type="model")
        # Download the artifact contents to a local directory
        artifact_dir = artifact.download()

        # The actual model file is inside the downloaded directory
        pipeline_path = os.path.join(artifact_dir, "toxic_comment_pipeline.pkl")
        pipeline = joblib.load(pipeline_path)
        logger.info("Production model pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from W&B. Error: {e}")
        # If the model fails to load, the pipeline will remain None,
        # and health checks will fail.
        pipeline = None
    finally:
        # Ensure the W&B run is finished
        if wandb.run:
            wandb.finish()


# --- DynamoDB Integration ---
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "prediction_logs")
try:
    # Boto3 will automatically use the IAM role credentials
    # when this code is running on an EC2 instance with the role attached.
    dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    # Perform a test operation to confirm connection
    table.load()
    logger.info(f"Successfully connected to DynamoDB table: {DYNAMODB_TABLE_NAME}")
except ClientError as e:
    logger.error(f"Failed to connect to DynamoDB: {e.response['Error']['Message']}")
    table = None # Set table to None if connection fails

# --- API Data Models (Pydantic) ---
# Defines the expected request JSON structure
class PredictionRequest(BaseModel):
    text: str

# Defines the response JSON structure
class PredictionResponse(BaseModel):
    prediction_id: str
    classification: str # "toxic" or "not_toxic"


# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check():
    """
    A simple endpoint to verify that the API is running,
    the model is loaded, and the database is connected.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable.")
    if table is None:
        raise HTTPException(status_code=503, detail="Database not connected or unavailable.")
    return {"status": "ok", "model_ready": True, "database_ready": True}


@app.post("/predict", response_model=PredictionResponse, summary="Classify a comment")
def predict(request: PredictionRequest):
    """
    Takes a text string and returns a classification of 'toxic' or 'not_toxic'.
    Each prediction is logged with a unique ID to DynamoDB.
    """
    if pipeline is None or table is None:
        raise HTTPException(status_code=503, detail="Service is not fully operational. Check health endpoint.")

    # 1. Generate a unique ID for this prediction event
    prediction_id = str(uuid.uuid4())

    # 2. Use the loaded pipeline to make a prediction
    # The pipeline expects an iterable (like a list)
    prediction_array = pipeline.predict([request.text])
    # The result is a numpy array, get the first element
    classification_result = "toxic" if prediction_array[0] == 1 else "not_toxic"

    # 3. Create the log item to be stored in DynamoDB
    log_item = {
        "prediction_id": prediction_id,
        "text_input": request.text,
        "classification": classification_result,
        "model_artifact_used": MODEL_ARTIFACT_PATH,
        "timestamp": pd.Timestamp.now().isoformat(), # Use ISO 8601 format
        "user_feedback": "N/A" # Default feedback status
    }

    # 4. Write the log item to the DynamoDB table
    try:
        table.put_item(Item=log_item)
        logger.info(f"Logged prediction {prediction_id} to DynamoDB.")
    except ClientError as e:
        # Log the error but don't fail the user's request
        logger.error(f"DynamoDB put_item failed: {e.response['Error']['Message']}")

    # 5. Return the result to the user
    return PredictionResponse(prediction_id=prediction_id, classification=classification_result)