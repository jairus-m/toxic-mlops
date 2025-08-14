"""
Streamlit Monitoring Dashboard for Toxic Comment Classification

This application provides monitoring and analytics for the toxic comment
classification model, including data drift analysis, model performance,
and user feedback insights.
"""

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.streamlit_monitoring.utils.data_loader import (
    load_feedback_logs,
    load_dataset,
    load_all_logs,
)
from src.core import logger

TOXICITY_TYPES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

st.set_page_config(
    page_title="Toxic Comment Model Monitoring", layout="wide"
)
logger.info("Streamlit toxic comment monitoring app started.")

st.title("Toxic Comment Model Monitoring Dashboard")
st.markdown(
    "Monitor model performance, data drift, and user feedback for the toxic comment classification system."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Refresh Data", use_container_width=True):
        logger.info("'Refresh Data' button clicked.")
        st.rerun()


@st.cache_data(ttl=5)
def load_data():
    """Load all monitoring data with caching"""
    try:
        logger.info("Loading all logs, feedback logs, and toxic comment dataset.")
        all_logs = load_all_logs()
        feedback_logs = load_feedback_logs()
        toxic_df = load_dataset()
        logger.info("Data loading complete.")
        return all_logs, feedback_logs, toxic_df
    except Exception as e:
        logger.exception(f"Failed to load data for monitoring dashboard: {e}")
        st.error(f"Failed to load data: {e}")
        return [], [], pd.DataFrame()


def parse_log_data(all_logs, feedback_logs):
    """Parse and structure log data for analysis"""
    # Parse prediction logs
    prediction_data = []
    for log in all_logs:
        if "probabilities" in log:
            row = {
                "timestamp": log.get("timestamp"),
                "endpoint": log.get("endpoint"),
                "text_length": len(log.get("request_text", "")),
                "is_toxic": log.get("is_toxic", False),
                "max_probability": log.get("max_probability", 0),
            }
            # Add individual probabilities
            for toxicity_type in TOXICITY_TYPES:
                row[f"{toxicity_type}_prob"] = log.get("probabilities", {}).get(
                    toxicity_type, 0
                )
                row[f"{toxicity_type}_pred"] = log.get("predictions", {}).get(
                    toxicity_type, False
                )
            prediction_data.append(row)

    # Parse feedback logs
    feedback_data = []
    for log in feedback_logs:
        if "predicted_labels" in log and "true_labels" in log:
            row = {
                "timestamp": log.get("timestamp"),
                "text_length": len(log.get("request_text", "")),
                "is_prediction_correct": log.get("is_prediction_correct", False),
                "user_comments": log.get("user_comments", ""),
            }
            # Add label comparisons
            predicted = log.get("predicted_labels", {})
            true = log.get("true_labels", {})
            for toxicity_type in TOXICITY_TYPES:
                row[f"{toxicity_type}_predicted"] = predicted.get(toxicity_type, False)
                row[f"{toxicity_type}_true"] = true.get(toxicity_type, False)
                row[f"{toxicity_type}_correct"] = predicted.get(
                    toxicity_type, False
                ) == true.get(toxicity_type, False)
            feedback_data.append(row)

    return pd.DataFrame(prediction_data), pd.DataFrame(feedback_data)


all_logs, feedback_logs, toxic_df = load_data()

if not all_logs:
    st.warning(
        "No prediction logs found. Please make some predictions to see the monitoring dashboard."
    )
    logger.warning("No prediction logs found for monitoring dashboard.")
else:
    prediction_df, feedback_df = parse_log_data(all_logs, feedback_logs)

    st.header("Overview Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_predictions = len(prediction_df)
        st.metric("Total Predictions", f"{total_predictions:,}")

    with col2:
        if total_predictions > 0:
            toxic_rate = prediction_df["is_toxic"].mean() * 100
            st.metric("Toxicity Rate", f"{toxic_rate:.1f}%")
        else:
            st.metric("Toxicity Rate", "N/A")

    with col3:
        total_feedback = len(feedback_df)
        st.metric("Feedback Received", f"{total_feedback:,}")

    with col4:
        if total_feedback > 0:
            accuracy_rate = feedback_df["is_prediction_correct"].mean() * 100
            st.metric("User Accuracy", f"{accuracy_rate:.1f}%")
        else:
            st.metric("User Accuracy", "N/A")

    st.header("Data Drift Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comment Length Distribution")
        if not toxic_df.empty and "comment_text" in toxic_df.columns:
            training_lengths = toxic_df["comment_text"].str.len().dropna()
            inference_lengths = prediction_df["text_length"].dropna()

            drift_data = pd.DataFrame(
                {
                    "Length": list(
                        training_lengths.sample(min(1000, len(training_lengths)))
                    )
                    + list(inference_lengths),
                    "Source": ["Training Data"] * min(1000, len(training_lengths))
                    + ["Inference"] * len(inference_lengths),
                }
            )

            chart = (
                alt.Chart(drift_data)
                .transform_density(
                    "Length", as_=["Length", "density"], groupby=["Source"]
                )
                .mark_area(opacity=0.6)
                .encode(
                    x=alt.X("Length:Q", title="Comment Length (characters)"),
                    y=alt.Y("density:Q", title="Density"),
                    color=alt.Color(
                        "Source:N", scale=alt.Scale(range=["#1f77b4", "#ff7f0e"])
                    ),
                )
                .properties(
                    title="Comment Length Distribution: Training vs Inference",
                    height=300,
                )
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Training dataset not available for drift analysis")

    with col2:
        st.subheader("Toxicity Distribution Over Time")
        if not prediction_df.empty:
            prediction_df["datetime"] = pd.to_datetime(
                prediction_df["timestamp"], errors="coerce"
            )
            prediction_df["date"] = prediction_df["datetime"].dt.date

            daily_stats = (
                prediction_df.groupby("date")
                .agg({"is_toxic": ["count", "sum", "mean"]})
                .round(3)
            )
            daily_stats.columns = ["Total", "Toxic_Count", "Toxic_Rate"]
            daily_stats = daily_stats.reset_index()

            if len(daily_stats) > 0:
                chart = (
                    alt.Chart(daily_stats)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y(
                            "Toxic_Rate:Q",
                            title="Toxicity Rate",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        tooltip=["date:T", "Total:Q", "Toxic_Count:Q", "Toxic_Rate:Q"],
                    )
                    .properties(title="Daily Toxicity Rate Trend", height=300)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Insufficient data for time trend analysis")

    st.header("Toxicity Type Analysis")

    if not prediction_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Frequency by Type")

            # Calculate prediction rates for each toxicity type
            type_stats = []
            for tox_type in TOXICITY_TYPES:
                pred_col = f"{tox_type}_pred"
                if pred_col in prediction_df.columns:
                    rate = prediction_df[pred_col].mean()
                    count = prediction_df[pred_col].sum()
                    type_stats.append(
                        {
                            "Toxicity_Type": tox_type.replace("_", " ").title(),
                            "Prediction_Rate": rate,
                            "Count": count,
                        }
                    )

            if type_stats:
                type_df = pd.DataFrame(type_stats)

                chart = (
                    alt.Chart(type_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Prediction_Rate:Q", title="Prediction Rate"),
                        y=alt.Y("Toxicity_Type:N", sort="-x", title="Toxicity Type"),
                        color=alt.Color(
                            "Prediction_Rate:Q", scale=alt.Scale(scheme="reds")
                        ),
                        tooltip=["Toxicity_Type:N", "Prediction_Rate:Q", "Count:Q"],
                    )
                    .properties(title="Toxicity Type Prediction Rates", height=300)
                )
                st.altair_chart(chart, use_container_width=True)

        with col2:
            st.subheader("Average Confidence by Type")

            # Calculate average probabilities
            prob_stats = []
            for tox_type in TOXICITY_TYPES:
                prob_col = f"{tox_type}_prob"
                if prob_col in prediction_df.columns:
                    avg_prob = prediction_df[prob_col].mean()
                    prob_stats.append(
                        {
                            "Toxicity_Type": tox_type.replace("_", " ").title(),
                            "Average_Probability": avg_prob,
                        }
                    )

            if prob_stats:
                prob_df = pd.DataFrame(prob_stats)

                fig = px.bar(
                    prob_df,
                    x="Average_Probability",
                    y="Toxicity_Type",
                    orientation="h",
                    title="Average Probability Scores by Type",
                    color="Average_Probability",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    st.header("Model Performance Analysis")

    if not feedback_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Overall Performance Metrics")

            overall_accuracy = feedback_df["is_prediction_correct"].mean()

            # Calculate per-label metrics
            label_metrics = []
            for tox_type in TOXICITY_TYPES:
                pred_col = f"{tox_type}_predicted"
                true_col = f"{tox_type}_true"

                if pred_col in feedback_df.columns and true_col in feedback_df.columns:
                    y_true = feedback_df[true_col]
                    y_pred = feedback_df[pred_col]

                    if len(y_true) > 0 and y_true.sum() > 0:  # Avoid division by zero
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)

                        label_metrics.append(
                            {
                                "Toxicity_Type": tox_type.replace("_", " ").title(),
                                "Accuracy": accuracy,
                                "Precision": precision,
                                "Recall": recall,
                                "F1_Score": f1,
                            }
                        )

            st.metric("Overall Accuracy", f"{overall_accuracy:.3f}")

            if label_metrics:
                metrics_df = pd.DataFrame(label_metrics)
                st.dataframe(
                    metrics_df.round(3), use_container_width=True, hide_index=True
                )

        with col2:
            st.subheader("Accuracy by Toxicity Type")

            if label_metrics:
                fig = px.bar(
                    metrics_df,
                    x="Toxicity_Type",
                    y="Accuracy",
                    title="Per-Label Accuracy Scores",
                    color="Accuracy",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1],
                )
                fig.update_layout(height=300, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

    st.header("Prediction Confidence Analysis")

    if not prediction_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confidence Distribution")

            # Max probability distribution
            if "max_probability" in prediction_df.columns:
                fig = px.histogram(
                    prediction_df,
                    x="max_probability",
                    nbins=20,
                    title="Distribution of Maximum Confidence Scores",
                    labels={
                        "max_probability": "Maximum Probability",
                        "count": "Frequency",
                    },
                )
                fig.add_vline(
                    x=0.5,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Decision Threshold",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Toxic vs Non-Toxic Confidence")

            if (
                "max_probability" in prediction_df.columns
                and "is_toxic" in prediction_df.columns
            ):
                conf_comparison = pd.DataFrame(
                    {
                        "Confidence": prediction_df["max_probability"],
                        "Prediction": prediction_df["is_toxic"].map(
                            {True: "Toxic", False: "Non-Toxic"}
                        ),
                    }
                )

                fig = px.box(
                    conf_comparison,
                    x="Prediction",
                    y="Confidence",
                    title="Confidence Scores by Prediction Type",
                    color="Prediction",
                    color_discrete_map={"Toxic": "#ff4444", "Non-Toxic": "#44ff44"},
                )
                st.plotly_chart(fig, use_container_width=True)

    st.header("Model Health Alerts")

    alerts = []

    # Check overall accuracy
    if not feedback_df.empty:
        overall_accuracy = feedback_df["is_prediction_correct"].mean()
        if overall_accuracy < 0.8:
            alerts.append(
                {
                    "type": "error",
                    "message": f"Overall model accuracy has dropped to {overall_accuracy:.2%} (below 80% threshold)",
                    "severity": "High",
                }
            )
        elif overall_accuracy < 0.9:
            alerts.append(
                {
                    "type": "warning",
                    "message": f"Overall model accuracy is {overall_accuracy:.2%} (below 90% target)",
                    "severity": "Medium",
                }
            )

    # Check prediction volume
    if not prediction_df.empty:
        recent_predictions = len(
            prediction_df[
                prediction_df["datetime"] > datetime.now() - timedelta(days=1)
            ]
        )
        if recent_predictions == 0:
            alerts.append(
                {
                    "type": "warning",
                    "message": "No predictions made in the last 24 hours",
                    "severity": "Medium",
                }
            )

    # Check toxicity rate
    if not prediction_df.empty:
        toxic_rate = prediction_df["is_toxic"].mean()
        if toxic_rate > 0.5:
            alerts.append(
                {
                    "type": "warning",
                    "message": f"High toxicity rate detected: {toxic_rate:.2%} of recent comments flagged as toxic",
                    "severity": "Medium",
                }
            )

    # Display alerts
    if alerts:
        for alert in alerts:
            if alert["type"] == "error":
                st.error(f"{alert['message']}")
                logger.warning(f"ALERT: {alert['message']}")
            else:
                st.warning(f"{alert['message']}")
                logger.info(f"WARNING: {alert['message']}")
    else:
        st.success("All systems operating normally")
        logger.info("No alerts detected - system operating normally")

    st.header("Raw User Feedback")

    if feedback_logs:
        feedback_data = []
        for log in feedback_logs:
            row = {
                "text": log.get("request_text", ""),
                "is_prediction_correct": log.get("is_prediction_correct", None)
            }

            # Add true labels from feedback
            true_labels = log.get("true_labels", {})
            for t_type in TOXICITY_TYPES:
                row[f"is_{t_type}"] = true_labels.get(t_type)
            
            feedback_data.append(row)
        
        feedback_display_df = pd.DataFrame(feedback_data)
        st.dataframe(feedback_display_df)
    else:
        st.info("No user feedback has been submitted yet.")


st.markdown("---")
st.markdown(
    f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"**Built with:** Streamlit, FastAPI, Scikit-learn"
)