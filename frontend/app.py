import streamlit as st
import requests
import os

# Get the API URL from the environment variable, with a default for local testing
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Toxic Comment Classifier")
st.markdown("Enter a comment below to classify it as toxic or not toxic.")

# Initialize session state for storing prediction results
if "prediction_id" not in st.session_state:
    st.session_state.prediction_id = None
    st.session_state.classification = None
    st.session_state.text_input = ""
    st.session_state.feedback_submitted = False


def submit_feedback(pred_id, feedback):
    """Sends user feedback to the backend API."""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={"prediction_id": pred_id, "feedback": feedback},
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        st.success("Thank you for your feedback!")
        st.session_state.feedback_submitted = True
    except requests.exceptions.RequestException as e:
        st.error(f"Error submitting feedback: {e}")


# Text input from user
comment_text = st.text_area("Enter your comment:", height=150)

if st.button("Classify Comment"):
    if comment_text:
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": comment_text})
            response.raise_for_status()
            result = response.json()

            # Store results in session state to persist across reruns
            st.session_state.prediction_id = result["prediction_id"]
            st.session_state.classification = result["classification"]
            st.session_state.text_input = comment_text
            st.session_state.feedback_submitted = False

        except requests.exceptions.RequestException as e:
            st.error(f"Error calling API: {e}")
            st.session_state.prediction_id = None
    else:
        st.warning("Please enter a comment to classify.")

# Display results and feedback buttons only after a prediction is made
if st.session_state.prediction_id:
    st.divider()
    st.markdown(f"**Original Comment:** *'{st.session_state.text_input}'*")

    if st.session_state.classification == "toxic":
        st.error(f"**Classification: {st.session_state.classification.upper()}**")
    else:
        st.success(f"**Classification: {st.session_state.classification.upper()}**")

    st.markdown(f"Prediction ID: `{st.session_state.prediction_id}`")

    # Show feedback buttons only if feedback has not been submitted for this prediction
    if not st.session_state.feedback_submitted:
        st.write("Was this classification correct?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Correct"):
                submit_feedback(st.session_state.prediction_id, "correct")
        with col2:
            if st.button("👎 Incorrect"):
                submit_feedback(st.session_state.prediction_id, "incorrect")
