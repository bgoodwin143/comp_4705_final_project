import streamlit as st
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

st.set_page_config(layout="wide")
st.title("Live Model Monitoring Dashboard")


# --- DynamoDB Connection ---
@st.cache_resource(ttl=60)  # Cache the resource for 60 seconds
def get_dynamodb_table():
    """Connects to DynamoDB and returns the table resource."""
    try:
        dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
        table = dynamodb.Table("prediction_logs")
        table.load()  # This will raise an error if the table doesn't exist
        return table
    except ClientError as e:
        st.error(f"Failed to connect to DynamoDB: {e.response['Error']['Message']}")
        return None


@st.cache_data(ttl=30)  # Cache the data for 30 seconds
def load_data(_table):
    """Scans the entire DynamoDB table and returns data as a DataFrame."""
    if _table is None:
        return pd.DataFrame()
    try:
        response = _table.scan()
        data = response.get("Items", [])
        # Handle pagination for large tables
        while "LastEvaluatedKey" in response:
            response = _table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            data.extend(response.get("Items", []))
        return pd.DataFrame(data)
    except ClientError as e:
        st.error(f"Failed to scan DynamoDB table: {e.response['Error']['Message']}")
        return pd.DataFrame()


table = get_dynamodb_table()
log_df = load_data(table)


if log_df.empty:
    st.warning("No prediction log data found in DynamoDB.")
else:
    # --- Data Cleaning and Preparation ---
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
    log_df["prediction_latency_ms"] = pd.to_numeric(
        log_df["prediction_latency_ms"], errors="coerce"
    )
    log_df = log_df.sort_values(by="timestamp", ascending=False)

    # --- Top Level Metrics ---
    st.header("Overall Performance")
    total_predictions = len(log_df)
    avg_latency = log_df["prediction_latency_ms"].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Predictions Made", f"{total_predictions}")
    col2.metric("Average Prediction Latency (ms)", f"{avg_latency:.2f}")

    st.divider()

    # --- Live Accuracy & Latency Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        st.header("Live Model Accuracy (Based on User Feedback)")
        feedback_df = log_df[log_df["user_feedback"] != "N/A"].copy()
        total_feedback = len(feedback_df)
        if total_feedback > 0:
            correct_feedback = (feedback_df["user_feedback"] == "correct").sum()
            live_accuracy = (correct_feedback / total_feedback) * 100
            st.metric(
                "Live Accuracy",
                f"{live_accuracy:.2f}%",
                f"Based on {total_feedback} feedback entries",
            )
        else:
            st.info("No user feedback has been submitted yet.")

    with col2:
        st.header("Prediction Latency Over Time")
        latency_chart_df = log_df[["timestamp", "prediction_latency_ms"]].copy()
        latency_chart_df = latency_chart_df.set_index("timestamp")
        st.line_chart(latency_chart_df)

    # --- Target Drift ---
    st.header("Target Drift: Distribution of Predictions")
    prediction_dist = log_df["classification"].value_counts()
    st.bar_chart(prediction_dist)

    # --- Recent Predictions Log ---
    st.header("Recent Prediction Logs")
    st.dataframe(log_df.head(10))
