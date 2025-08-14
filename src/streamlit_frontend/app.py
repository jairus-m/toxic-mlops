"""
Streamlit Frontend for Toxic Comment Classification
"""

import os
import requests
import streamlit as st
from src.core import logger

FASTAPI_BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000")

TOXICITY_TYPES = {
    "toxic": "A broad category of offensive or inappropriate content.",
    "severe_toxic": "Content that is extremely abusive, insulting, or threatening.",
    "obscene": "Language that is vulgar, lewd, or sexually explicit.",
    "threat": "Direct or indirect statements of intent to inflict harm.",
    "insult": "Content that is disrespectful, offensive, or demeaning to an individual or group.",
    "identity_hate": "Speech that attacks or demeans a group based on identity (e.g., race, religion, sexual orientation).",
}

st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")
logger.info("Streamlit toxic comment classification frontend app started.")

# Session state variables
if "comment_text" not in st.session_state:
    st.session_state.comment_text = ""
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


def handle_feedback(user_labels: dict, user_comments: str = ""):
    """
    Sends feedback to the backend.
    """
    if st.session_state.prediction_result:
        predicted_labels = st.session_state.prediction_result["predictions"]
        predicted_probabilities = st.session_state.prediction_result["probabilities"]

        is_correct = predicted_labels == user_labels

        feedback_payload = {
            "request_text": st.session_state.comment_text,
            "predicted_labels": predicted_labels,
            "predicted_probabilities": predicted_probabilities,
            "true_labels": user_labels,
            "is_prediction_correct": is_correct,
            "user_comments": user_comments,
        }

        try:
            logger.info(f"Submitting feedback: {feedback_payload}")
            response = requests.post(
                f"{FASTAPI_BACKEND_URL}/feedback", json=feedback_payload
            )
            response.raise_for_status()
            st.session_state.feedback_submitted = True
            st.toast("Thank you for your feedback!")
            logger.info("Feedback submitted successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not submit feedback: {e}")
            logger.error(f"Could not submit feedback to backend: {e}", exc_info=True)


if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# --- Header ---
st.title("Toxic Comment Classifier")
st.markdown(
    """
    This application analyzes text to detect different types of toxicity. 
    It is designed to help identify and filter harmful content in online discussions, social media, and other platforms.
    The tool uses a machine learning model to classify comments into six categories of toxicity.
    """
)

# --- Toxicity Descriptions ---
with st.expander("Toxicity Type Descriptions"):
    for key, desc in TOXICITY_TYPES.items():
        st.markdown(f"**{key.replace('_', ' ').title()}**: {desc}")

# --- Main Input Section ---
st.subheader("Enter Comment to Analyze")

if st.button("Get Random Example Comment"):
    logger.info("'Get Random Example Comment' button clicked.")
    try:
        response = requests.get(f"{FASTAPI_BACKEND_URL}/example")
        response.raise_for_status()
        example_data = response.json()
        example_comment = example_data.get("comment", "")
        st.session_state.comment_text = example_comment
        st.session_state.prediction_result = None
        st.session_state.feedback_submitted = False
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend: {e}")
        st.warning("Please ensure the FastAPI backend service is running.")
        logger.error(
            f"Could not connect to backend for random comment: {e}", exc_info=True
        )

comment_text = st.text_area(
    "Comment:",
    height=150,
    key="comment_text",
    placeholder="Enter a comment here to analyze for toxicity...",
    help="Enter any text comment to check for various types of toxicity",
)

if st.button("Analyze Comment", type="primary"):
    if not st.session_state.comment_text.strip():
        st.warning("Please enter a comment before analyzing.")
    else:
        try:
            with st.spinner("Analyzing comment for toxicity..."):
                payload = {"text": st.session_state.comment_text}
                response = requests.post(
                    f"{FASTAPI_BACKEND_URL}/predict_proba", json=payload
                )
                response.raise_for_status()
                st.session_state.prediction_result = response.json()
                st.session_state.feedback_submitted = False
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {e}")
            st.info(f"Please ensure the backend is running at `{FASTAPI_BACKEND_URL}`.")
            st.session_state.prediction_result = None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.prediction_result = None
            logger.exception(f"Unexpected error during toxicity analysis: {e}")

# --- Display Results ---
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    is_toxic = result.get("is_toxic", False)
    predictions = result.get("predictions", {})
    probabilities = result.get("probabilities", {})
    max_prob = result.get("max_toxicity_probability", 0)

    st.subheader("Analysis Results")
    if is_toxic:
        st.error(f"TOXIC CONTENT DETECTED (Max confidence: {max_prob * 100:.1f}%)")
    else:
        st.success("This comment appears to be non-toxic.")

    tab1, tab2, tab3 = st.tabs(
        ["Detailed Results", "Provide Feedback", "Moderation Actions"]
    )

    with tab1:
        st.write("**Detailed Classification Results:**")
        col1, col2, col3 = st.columns(3)
        for i, (toxicity_type, is_predicted) in enumerate(predictions.items()):
            prob = probabilities.get(toxicity_type, 0)
            target_col = [col1, col2, col3][i % 3]
            with target_col:
                st.markdown(f"**{toxicity_type.replace('_', ' ').title()}**")
                st.markdown(f"Confidence of being toxic: {prob * 100:.1f}%")
                st.markdown("---")

    with tab2:
        if not st.session_state.feedback_submitted:
            st.write("**Help improve the model by providing feedback:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Prediction is Correct", use_container_width=True):
                    handle_feedback(predictions, "User confirmed prediction is correct")
            with col2:
                if st.button("Prediction is Wrong", use_container_width=True):
                    corrected_labels = {k: not v for k, v in predictions.items()}
                    handle_feedback(
                        corrected_labels,
                        "User indicated prediction is wrong - labels inverted",
                    )
            st.markdown("---")
            with st.expander("Provide Detailed Feedback"):
                corrected_labels = {}
                for toxicity_type in predictions.keys():
                    corrected_labels[toxicity_type] = st.checkbox(
                        f"{toxicity_type.replace('_', ' ').title()}",
                        value=predictions[toxicity_type],
                        key=f"feedback_{toxicity_type}",
                    )
                user_comments = st.text_area(
                    "Additional feedback (optional):",
                    placeholder="Any additional comments...",
                    height=100,
                )
                if st.button("Submit Detailed Feedback", type="primary"):
                    handle_feedback(corrected_labels, user_comments)
        else:
            st.success("Thank you for your feedback!")

    with tab3:
        st.write("**Moderation Actions:**")

        # Show moderation recommendation based on toxicity
        if is_toxic:
            st.warning("⚠️ This content contains potential toxicity")

            moderation_payload = {
                "text": st.session_state.comment_text,
                "context": "streamlit_frontend",
                "user_id": None,
            }

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "🟢 Allow Content",
                    use_container_width=True,
                    help="Allow this content to be published",
                ):
                    try:
                        with st.spinner("Processing moderation decision..."):
                            response = requests.post(
                                f"{FASTAPI_BACKEND_URL}/moderate",
                                json=moderation_payload,
                            )
                            response.raise_for_status()
                            moderation_result = response.json()

                            if moderation_result.get("decision") == "allow":
                                st.success("✅ Content allowed!")
                            else:
                                st.info(
                                    f"🤖 Auto-moderation decision: {moderation_result.get('decision')}"
                                )
                                st.write(
                                    f"Reasoning: {moderation_result.get('reasoning', 'No reasoning provided')}"
                                )

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error processing moderation: {e}")
                        logger.error(f"Moderation API error: {e}")

            with col2:
                if st.button(
                    "🔴 Block Content",
                    use_container_width=True,
                    help="Block this content from being published",
                ):
                    try:
                        with st.spinner("Processing moderation decision..."):
                            response = requests.post(
                                f"{FASTAPI_BACKEND_URL}/moderate",
                                json=moderation_payload,
                            )
                            response.raise_for_status()
                            moderation_result = response.json()

                            st.error("🚫 Content blocked!")
                            st.write(
                                f"Auto-moderation reasoning: {moderation_result.get('reasoning', 'Manual block decision')}"
                            )

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error processing moderation: {e}")
                        logger.error(f"Moderation API error: {e}")

            with col3:
                if st.button(
                    "⚡ Auto-Moderate",
                    use_container_width=True,
                    help="Let the AI decide the moderation action",
                ):
                    try:
                        with st.spinner("Getting AI moderation decision..."):
                            response = requests.post(
                                f"{FASTAPI_BACKEND_URL}/moderate",
                                json=moderation_payload,
                            )
                            response.raise_for_status()
                            moderation_result = response.json()

                            decision = moderation_result.get("decision")
                            confidence = moderation_result.get("confidence", 0)
                            reasoning = moderation_result.get(
                                "reasoning", "No reasoning provided"
                            )

                            if decision == "allow":
                                st.success(
                                    f"✅ AI Decision: Allow (Confidence: {confidence:.2f})"
                                )
                            elif decision == "block":
                                st.error(
                                    f"🚫 AI Decision: Block (Confidence: {confidence:.2f})"
                                )
                            elif decision == "human_review":
                                st.warning(
                                    f"👥 AI Decision: Needs Human Review (Confidence: {confidence:.2f})"
                                )
                                queue_id = moderation_result.get("queue_id")
                                if queue_id:
                                    st.info(f"📋 Queued for review: `{queue_id}`")

                            st.write(f"**Reasoning:** {reasoning}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Error getting AI moderation decision: {e}")
                        logger.error(f"Moderation API error: {e}")

            # Show moderation queue info if available
            try:
                queue_response = requests.get(f"{FASTAPI_BACKEND_URL}/moderation-stats")
                if queue_response.status_code == 200:
                    queue_stats = queue_response.json().get("queue_statistics", {})
                    pending_items = queue_stats.get("pending_items", 0)

                    if pending_items > 0:
                        st.markdown("---")
                        st.info(
                            f"📊 Current moderation queue: {pending_items} items pending human review"
                        )

                        if st.button("View Moderation Queue", use_container_width=True):
                            queue_response = requests.get(
                                f"{FASTAPI_BACKEND_URL}/moderation-queue?limit=10"
                            )
                            if queue_response.status_code == 200:
                                queue_data = queue_response.json()
                                items = queue_data.get("items", [])

                                if items:
                                    st.write("**Recent items in moderation queue:**")
                                    for item in items[:5]:  # Show top 5
                                        with st.expander(
                                            f"Queue ID: {item.get('queue_id', 'Unknown')} ({item.get('priority', 'medium')} priority)"
                                        ):
                                            st.write(
                                                f"**Text:** {item.get('text', '')[:200]}..."
                                            )
                                            st.write(
                                                f"**Created:** {item.get('created_at', 'Unknown')}"
                                            )

                                            # Show predicted toxicity types
                                            predictions = item.get(
                                                "toxicity_predictions", {}
                                            )
                                            toxic_types = [
                                                k for k, v in predictions.items() if v
                                            ]
                                            if toxic_types:
                                                st.write(
                                                    f"**Detected toxicity:** {', '.join(toxic_types)}"
                                                )
                                else:
                                    st.info(
                                        "No items currently in the moderation queue"
                                    )

            except requests.exceptions.RequestException:
                pass  # Silently fail if moderation stats not available

        else:
            st.info("ℹ️ This content appears safe - no moderation action needed")
            st.write(
                "Content classification shows low toxicity probability across all categories."
            )

# --- Footer ---
st.markdown("---")
st.subheader("About")
st.markdown(
    "Built with Streamlit, FastAPI, and Scikit-learn for toxic comment detection."
)
