"""
Streamlit Frontend for Toxic Comment Classification

This application provides a user interface to interact with the
FastAPI toxic comment classification backend.
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go
from src.core import logger

FASTAPI_BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000")

TOXICITY_TYPES = {
    "toxic": "General toxicity",
    "severe_toxic": "Severely toxic content",
    "obscene": "Obscene language",
    "threat": "Threatening language",
    "insult": "Insulting content",
    "identity_hate": "Identity-based hate speech",
}

st.set_page_config(page_title="Toxic Comment Classifier", layout="wide", page_icon="🛡️")
logger.info("Streamlit toxic comment classification frontend app started.")

# Session state variables for ML monitoring
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


def create_toxicity_chart(probabilities: dict, predictions: dict):
    """
    Create a bar chart showing toxicity probabilities
    """
    labels = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ["red" if predictions[label] else "lightblue" for label in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=probs,
                marker_color=colors,
                text=[f"{p:.3f}" for p in probs],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Toxicity Probabilities by Type",
        xaxis_title="Toxicity Type",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False,
    )

    return fig


# --- Header ---
st.title("🛡️ Toxic Comment Classifier")
st.markdown(
    "Enter a comment below to analyze it for various types of toxicity. "
    "This app uses machine learning to detect toxic, threatening, insulting, and other harmful content."
)

# --- Sidebar Model Info ---
with st.sidebar:
    st.header("📊 Model Information")
    try:
        stats_response = requests.get(f"{FASTAPI_BACKEND_URL}/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            st.write("**Model Type:** Multi-label Toxicity Classifier")

            if "training_metrics" in stats:
                metrics = stats["training_metrics"]
                st.write(
                    f"**Mean ROC-AUC:** {metrics.get('test_mean_roc_auc', 'N/A'):.4f}"
                )
                st.write(
                    f"**Exact Match Accuracy:** {metrics.get('test_exact_match_accuracy', 'N/A'):.4f}"
                )

                if "per_label_metrics" in metrics:
                    st.write("**Per-Label ROC-AUC:**")
                    for label, metric in metrics["per_label_metrics"].items():
                        st.write(f"  • {label}: {metric.get('roc_auc', 0):.3f}")
        else:
            st.write("Model stats unavailable")
    except Exception as e:
        st.write("Could not fetch model stats")
        logger.error(f"Error fetching model stats: {e}")

    st.markdown("---")
    st.write("**Toxicity Types:**")
    for key, desc in TOXICITY_TYPES.items():
        st.write(f"• **{key.title()}**: {desc}")

# --- Main Input Section ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Enter Comment to Analyze")

    if st.button("🎲 Get Random Example Comment"):
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

    if st.button("🔍 Analyze Comment", type="primary"):
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
                st.info(
                    f"Please ensure the backend is running at `{FASTAPI_BACKEND_URL}`."
                )
                st.session_state.prediction_result = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.prediction_result = None
                logger.exception(f"Unexpected error during toxicity analysis: {e}")

with col2:
    st.subheader("ℹ️ About This Tool")
    st.markdown("""
    This classifier detects 6 types of toxicity:
    
    🔴 **Toxic**: General harmful content  
    🟠 **Severe Toxic**: Extremely harmful content  
    🟡 **Obscene**: Inappropriate language  
    🔵 **Threat**: Threatening language  
    🟣 **Insult**: Insulting content  
    🟤 **Identity Hate**: Hate based on identity  
    """)

# --- Display Results ---
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    is_toxic = result.get("is_toxic", False)
    predictions = result.get("predictions", {})
    probabilities = result.get("probabilities", {})
    max_prob = result.get("max_toxicity_probability", 0)

    st.subheader("🎯 Analysis Results")
    if is_toxic:
        st.error(
            f"⚠️ **TOXIC CONTENT DETECTED** (Max confidence: {max_prob * 100:.1f}%)"
        )
    else:
        st.success("✅ **NON-TOXIC CONTENT** - This comment appears to be safe")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Detailed Results", "📈 Probability Chart", "🔧 Provide Feedback"]
    )

    with tab1:
        st.write("**Detailed Classification Results:**")
        col1, col2, col3 = st.columns(3)
        for i, (toxicity_type, is_predicted) in enumerate(predictions.items()):
            prob = probabilities.get(toxicity_type, 0)
            target_col = [col1, col2, col3][i % 3]
            with target_col:
                if is_predicted:
                    st.markdown(f"🔴 **{toxicity_type.replace('_', ' ').title()}**")
                    st.markdown(f"Confidence: {prob * 100:.1f}%")
                else:
                    st.markdown(f"✅ **{toxicity_type.replace('_', ' ').title()}**")
                    st.markdown(f"Confidence: {(1 - prob) * 100:.1f}%")
                st.markdown("---")

    with tab2:
        fig = create_toxicity_chart(probabilities, predictions)
        st.plotly_chart(fig, use_container_width=True)
        prob_df = [
            {
                "Toxicity Type": toxicity_type.replace("_", " ").title(),
                "Probability": f"{prob:.4f}",
                "Prediction": "🔴 Toxic" if predictions[toxicity_type] else "✅ Safe",
            }
            for toxicity_type, prob in probabilities.items()
        ]
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with tab3:
        if not st.session_state.feedback_submitted:
            st.write("**Help improve the model by providing feedback:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Prediction is Correct", use_container_width=True):
                    handle_feedback(predictions, "User confirmed prediction is correct")
            with col2:
                if st.button("❌ Prediction is Wrong", use_container_width=True):
                    corrected_labels = {k: not v for k, v in predictions.items()}
                    handle_feedback(
                        corrected_labels,
                        "User indicated prediction is wrong - labels inverted",
                    )
            st.markdown("---")
            with st.expander("🔧 Provide Detailed Feedback"):
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
                if st.button("📤 Submit Detailed Feedback", type="primary"):
                    handle_feedback(corrected_labels, user_comments)
        else:
            st.success("✅ Thank you for your feedback!")

# --- Backend health check ---
try:
    health_response = requests.get(f"{FASTAPI_BACKEND_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("🟢 Backend: Online")
    else:
        st.sidebar.error("🔴 Backend: Error")
except Exception:
    st.sidebar.error("🔴 Backend: Offline")

# --- Footer ---
st.markdown("---")
if FASTAPI_BACKEND_URL != "http://localhost:8000":
    LOCAL_URL = "http://localhost:8000"
    st.markdown(f"🔗 Backend running at: `{FASTAPI_BACKEND_URL}`")
else:
    st.markdown(f"🔗 Backend running at: `{FASTAPI_BACKEND_URL}`")


# --- How This Works ---
with st.expander("ℹ️ How This Works"):
    st.markdown("""
    ### About the Toxic Comment Classifier
    
    This tool uses machine learning to analyze text comments and detect various types of toxicity.
    """)


# --- Callback for sample comments ---
def load_and_analyze_sample(sample):
    st.session_state.comment_text = sample
    st.session_state.prediction_result = None
    st.session_state.feedback_submitted = False
    try:
        payload = {"text": sample}
        with st.spinner("Analyzing sample comment for toxicity..."):
            response = requests.post(
                f"{FASTAPI_BACKEND_URL}/predict_proba", json=payload
            )
            response.raise_for_status()
            st.session_state.prediction_result = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {e}")
        st.info(f"Ensure backend is running at `{FASTAPI_BACKEND_URL}`.")
        st.session_state.prediction_result = None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state.prediction_result = None
        logger.exception(f"Unexpected error: {e}")


# --- Sample comments section ---
with st.expander("🧪 Sample Comments for Testing"):
    sample_comments = [
        "This is a really helpful and informative article. Thank you for sharing!",
        "You are completely wrong and have no idea what you're talking about.",
        "I disagree with your opinion, but I respect your right to express it.",
        "This is stupid and you're an idiot for posting this garbage.",
        "Great work on this project! Very impressive results.",
    ]
    st.write("Click any sample comment to test:")
    for i, sample in enumerate(sample_comments):
        st.button(
            f"📝 Sample {i + 1}: {sample[:50]}...",
            key=f"sample_{i}",
            on_click=load_and_analyze_sample,
            args=(sample,),
        )

st.markdown("---")
st.markdown(
    "Built with Streamlit, FastAPI, and Scikit-learn for toxic comment detection."
)
