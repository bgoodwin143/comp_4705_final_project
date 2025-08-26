# api/test_main.py
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
import boto3
import os
from api.main import app, get_pipeline, get_db_table

# Set dummy AWS credentials for Moto
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-2"

DYNAMODB_TABLE_NAME = "prediction_logs"


# --- Mock Objects and Override Functions ---
class MockPipeline:
    def predict(self, texts):
        return [1]  # Always predict 'toxic'


def override_get_pipeline():
    return MockPipeline()


@pytest.fixture
def mock_dynamodb_table():
    with mock_aws():
        dynamodb = boto3.resource(
            "dynamodb", region_name=os.environ["AWS_DEFAULT_REGION"]
        )
        table = dynamodb.create_table(
            TableName=DYNAMODB_TABLE_NAME,
            KeySchema=[{"AttributeName": "prediction_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "prediction_id", "AttributeType": "S"}
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
        yield table


def override_get_db_table(mock_dynamodb_table):
    # This is a bit tricky: we define a function that returns another function
    # so that pytest can inject the fixture.
    def inner():
        return mock_dynamodb_table

    return inner


# --- Test Functions ---
def test_health_check(mock_dynamodb_table):
    # Apply the overrides for the duration of this test
    app.dependency_overrides[get_pipeline] = override_get_pipeline
    app.dependency_overrides[get_db_table] = override_get_db_table(mock_dynamodb_table)

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_ready": True,
        "database_ready": True,
    }

    # Clear the overrides after the test
    app.dependency_overrides = {}


def test_predict_endpoint(mock_dynamodb_table):
    # Apply the overrides
    app.dependency_overrides[get_pipeline] = override_get_pipeline
    app.dependency_overrides[get_db_table] = override_get_db_table(mock_dynamodb_table)

    client = TestClient(app)
    payload = {"text": "this is a test comment"}
    response = client.post("/predict", json=payload)

    # Assert API response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["classification"] == "toxic"

    # Assert DynamoDB log
    prediction_id = response_data["prediction_id"]
    item = mock_dynamodb_table.get_item(Key={"prediction_id": prediction_id})
    assert "Item" in item
    assert item["Item"]["text_input"] == "this is a test comment"

    # Clear overrides
    app.dependency_overrides = {}
