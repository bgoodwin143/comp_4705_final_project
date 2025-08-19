import streamlit as st
import requests
import os

# It's good practice to get the API URL from an environment variable
# We'll set a default for local testing.
API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"

st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")

st.title("Toxic Comment Classifier")
st.write(
    "Enter a comment below to classify it as toxic or not toxic. "
    "This interface sends a request to a live FastAPI backend."
)

# --- User Input Area ---
with st.form("prediction_form"):
    text_input = st.text_area(
        "Enter your comment here:",
        "This is an example of a perfectly nice and respectful comment.",
        height=150,
    )
    submit_button = st.form_submit_button("Classify Comment")

# --- Prediction Logic ---
if submit_button and text_input:
    with st.spinner("Classifying..."):
        try:
            # The request body must match the Pydantic model in the API
            payload = {"text": text_input}

            # Send the request to the FastAPI backend
            response = requests.post(PREDICT_ENDPOINT, json=payload)

            if response.status_code == 200:
                # Get the JSON response
                prediction_data = response.json()
                classification = prediction_data.get("classification")
                prediction_id = prediction_data.get("prediction_id")

                # Display the result
                st.success(f"Classification Result: **{classification.upper()}**")
                st.info(
                    f"Prediction ID: `{prediction_id}` (This has been logged for monitoring)"
                )
            else:
                # Handle API errors
                st.error(f"Error: API returned status code {response.status_code}")
                st.json(response.json())

        except requests.exceptions.RequestException as e:
            st.error(f"Error: Could not connect to the API. Details: {e}")
