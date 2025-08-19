import streamlit as st
import pandas as pd
import boto3
from botocore.exceptions import ClientError
import plotly.express as px
import os

# --- AWS and DynamoDB Configuration ---
# For local testing, you might need to configure credentials.
# When deployed on EC2 with an IAM role, Boto3 handles this automatically.
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "prediction_logs")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")
st.title("Live Model Monitoring Dashboard")
st.write(
    "This dashboard connects to a DynamoDB table to visualize live prediction data."
)


# --- Data Loading Function ---
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def load_data_from_dynamodb():
    """
    Scans the entire DynamoDB table and returns the data as a Pandas DataFrame.
    """
    try:
        dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        # Scan is okay for smaller tables, but for very large tables,
        # you'd want a more efficient query strategy.
        response = table.scan()
        data = response.get("Items", [])

        # Handle paginated results if the table is large
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            data.extend(response.get("Items", []))

        if not data:
            return pd.DataFrame()  # Return empty DataFrame if no data

        df = pd.DataFrame(data)
        # Convert timestamp to datetime objects for plotting
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except ClientError as e:
        st.error(f"Error connecting to DynamoDB: {e.response['Error']['Message']}")
        return pd.DataFrame()  # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


# --- Main Dashboard Logic ---
df = load_data_from_dynamodb()

if df.empty:
    st.warning(
        "No data found in the prediction_logs table. Please send predictions from the API."
    )
else:
    st.success(f"Successfully loaded {len(df)} records from DynamoDB.")
    st.dataframe(df.head())

    st.divider()

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        st.header("Prediction Distribution (Target Drift)")
        # Create a bar chart of the 'classification' counts
        fig_dist = px.bar(
            df["classification"].value_counts().reset_index(),
            x="classification",
            y="count",
            title="Distribution of Predictions",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.header("Predictions Over Time")
        # Resample data to count predictions per hour
        predictions_over_time = (
            df.set_index("timestamp").resample("h").size().reset_index(name="count")
        )
        fig_time = px.line(
            predictions_over_time,
            x="timestamp",
            y="count",
            title="Prediction Volume (per hour)",
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.divider()
    st.header("Recent Prediction Logs")
    # Show the 10 most recent predictions
    st.dataframe(df.sort_values(by="timestamp", ascending=False).head(10))
