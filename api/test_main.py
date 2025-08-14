import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
import boto3
import os
from unittest.mock import patch

# --- Mocking Setup ---
# Set up dummy AWS credentials for Moto
# This is required for the mock_aws decorator to work.
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1" # Specify a default region

# Use a patch to prevent the app from trying to download the model from W&B during tests.
# We will "monkeypatch" the load_production_model function.
# This is an advanced but essential technique for unit testing.
with patch("api.main.load_production_model") as mock_load_model:
    from api.main import app, DYNAMODB_TABLE_NAME

# --- Test Fixtures (Reusable Components) ---
@pytest.fixture
def test_client():
    """
    Creates a FastAPI TestClient.
    """
    return TestClient(app)

@pytest.fixture
def mock_dynamodb_table():
    """
    Creates a mock DynamoDB table using Moto.
    This fixture runs within the 'mock_aws' context manager, so any boto3
    calls are intercepted and sent to a fake, in-memory DynamoDB.
    """
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name=os.environ["AWS_DEFAULT_REGION"])
        table = dynamodb.create_table(
            TableName=DYNAMODB_TABLE_NAME,
            KeySchema=[{"AttributeName": "prediction_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "prediction_id", "AttributeType": "S"}],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
        yield table # Provide the table to the test function

# --- Tests ---
def test_health_check(test_client):
    """
    Tests the /health endpoint.
    We need to manually set the global 'pipeline' and 'table' variables
    because the startup event is mocked and doesn't run.
    """
    app.pipeline = "mock_pipeline" # Simulate a loaded model
    app.table = "mock_table" # Simulate a connected table
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": True, "database_ready": True}
    # Reset globals after test
    app.pipeline = None
    app.table = None

def test_predict_endpoint(test_client, mock_dynamodb_table):
    """
    Tests the /predict endpoint for a successful prediction and logging.
    This test uses both the FastAPI test client and the mock DynamoDB table.
    """
    # Mock the scikit-learn pipeline's predict method to return a known value.
    # This isolates the test to only the API logic, not the model's performance.
    class MockPipeline:
        def predict(self, texts):
            return [1] # Always predict 'toxic'

    # Manually set the global variables for the scope of this test
    app.pipeline = MockPipeline()
    app.table = mock_dynamodb_table

    # Define the request payload
    payload = {"text": "this is a test comment"}

    # Make the request
    response = test_client.post("/predict", json=payload)

    # Assertions for the API response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["classification"] == "toxic"
    assert "prediction_id" in response_data

    # Assertions for the DynamoDB log
    prediction_id = response_data["prediction_id"]
    item = mock_dynamodb_table.get_item(Key={"prediction_id": prediction_id})
    assert "Item" in item
    db_item = item["Item"]
    assert db_item["text_input"] == "this is a test comment"
    assert db_item["classification"] == "toxic"

    # Reset globals after the test
    app.pipeline = None
    app.table = None

def test_predict_model_not_loaded(test_client):
    """
    Tests that the /predict endpoint returns a 503 error if the model is not loaded.
    """
    app.pipeline = None # Ensure pipeline is not loaded
    app.table = "mock_table" # DB can be connected
    response = test_client.post("/predict", json={"text": "test"})
    assert response.status_code == 503
    assert "Service is not fully operational" in response.json()["detail"]
    app.table = None