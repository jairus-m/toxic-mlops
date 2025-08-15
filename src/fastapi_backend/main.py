"""
FastAPI Toxic Comment Classification API

This API provides endpoints for toxicity analysis of text comments.
It is environment-aware and can load assets from local disk or S3.
"""

from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Response, Depends
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd
import time
from src.core import (
    logger,
    get_asset_path,
    prediction_logger,
)
from src.fastapi_backend.utils.middleware import (
    log_middleware_request,
    log_middleware_response,
)
from src.fastapi_backend.utils.schemas import (
    PredictRequest,
    ToxicityResponse,
    ToxicityProbabilityResponse,
    ExampleResponse,
    ToxicityFeedback,
    ModerationRequest,
    ModerationResponse,
    ReviewDecision,
)
from src.fastapi_backend.utils.model_loader import load_model
from src.fastapi_backend.utils.moderation import (
    apply_moderation_rules,
    queue_for_review,
    get_moderation_queue,
    process_review_decision,
    get_queue_stats,
)

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

app = FastAPI(
    title="Toxic Comment Classification API",
    description="API for detecting toxicity in text comments using multi-label classification",
    version="1.0.0",
)

# Middleware to log requests and responses
app.add_middleware(BaseHTTPMiddleware, dispatch=log_middleware_request)
app.add_middleware(BaseHTTPMiddleware, dispatch=log_middleware_response)

logger.info("FastAPI Toxic Comment Classification App initialized successfully!")


def get_model():
    """Dependency to get the ML model"""
    return load_model()


def clean_text(text: str) -> str:
    """
    Basic text cleaning function for comment preprocessing.
    Should match the preprocessing used during training.
    """
    import re

    if not text or pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())

    return text


@app.get("/")
async def root() -> dict:
    """
    Root endpoint
    Returns:
        dict: A dictionary with a message indicating the API is running
    """
    return {
        "message": "FastAPI Toxic Comment Classification API",
        "endpoints": [
            "/health",
            "/predict",
            "/predict_proba",
            "/example",
            "/feedback",
            "/moderate",
            "/moderation-queue",
        ],
        "toxicity_types": TARGET_COLS,
    }


@app.get("/health")
async def health_check(model=Depends(get_model)) -> dict:
    """
    Health check endpoint
    Returns:
        dict: A dictionary with the health status of the API
    """
    try:
        # Check if model has required methods
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert callable(model.predict)
        assert callable(model.predict_proba)

        # Test prediction on sample text
        test_text = "This is a test comment"
        cleaned_test = clean_text(test_text)
        predictions = model.predict([cleaned_test])
        probabilities = model.predict_proba([cleaned_test])

        # Verify output shapes
        assert predictions.shape[1] == len(TARGET_COLS)
        assert len(probabilities) == len(TARGET_COLS)

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_ready": True,
            "toxicity_labels": TARGET_COLS,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/predict")
async def predict(
    request: PredictRequest, model=Depends(get_model)
) -> ToxicityResponse:
    """
    Predict toxicity endpoint - returns binary predictions for each toxicity type
    Args:
        request (PredictRequest):  {"text": "string"}
    Returns:
        ToxicityResponse object with binary predictions
    """
    try:
        cleaned_text = clean_text(request.text)

        start_time = time.time()
        predictions = model.predict([cleaned_text])[0]  # Get first (and only) result
        end_time = time.time()
        latency = end_time - start_time

        toxicity_predictions = {}
        for i, label in enumerate(TARGET_COLS):
            toxicity_predictions[label] = bool(predictions[i])

        is_toxic = any(toxicity_predictions.values())

        prediction_log = {
            "endpoint": "/predict",
            "request_text": request.text[:100] + "..."
            if len(request.text) > 100
            else request.text,
            "prediction_latency": latency,
            "is_toxic": is_toxic,
            "predictions": toxicity_predictions,
        }
        prediction_logger.info(prediction_log)

        return {"is_toxic": is_toxic, "predictions": toxicity_predictions}

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error making prediction")


@app.post("/predict_proba")
async def predict_proba(
    request: PredictRequest, model=Depends(get_model)
) -> ToxicityProbabilityResponse:
    """
    Predict toxicity with probabilities endpoint
    Args:
        request (PredictRequest):  {"text": "string"}
    Returns:
        ToxicityProbabilityResponse object with probability scores
    """
    try:
        cleaned_text = clean_text(request.text)

        start_time = time.time()
        binary_predictions = model.predict([cleaned_text])[0]
        probability_predictions = model.predict_proba([cleaned_text])
        end_time = time.time()
        latency = end_time - start_time

        probabilities = {}
        binary_results = {}

        for i, label in enumerate(TARGET_COLS):
            prob_toxic = float(probability_predictions[i][0, 1])
            probabilities[label] = round(prob_toxic, 4)
            binary_results[label] = bool(binary_predictions[i])

        is_toxic = any(binary_results.values())
        max_toxicity_prob = max(probabilities.values())

        prediction_log = {
            "endpoint": "/predict_proba",
            "request_text": request.text[:100] + "..."
            if len(request.text) > 100
            else request.text,
            "prediction_latency": latency,
            "is_toxic": is_toxic,
            "max_probability": max_toxicity_prob,
            "probabilities": probabilities,
        }
        prediction_logger.info(prediction_log)

        return {
            "is_toxic": is_toxic,
            "max_toxicity_probability": round(max_toxicity_prob, 4),
            "predictions": binary_results,
            "probabilities": probabilities,
        }

    except ValueError as e:
        logger.error(f"Pydantic validation error: {str(e)}")
        raise HTTPException(status_code=422, detail="Invalid input format")
    except Exception as e:
        logger.error(f"Error making prediction with probabilities: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error making prediction with probabilities"
        )


@app.get("/example")
async def example() -> ExampleResponse:
    """
    Example endpoint to get a random comment from the dataset
    Returns:
        ExampleResponse object
    """
    try:
        data_path = get_asset_path("data")
        df = pd.read_csv(data_path)

        random_comment = df.sample(n=1)["comment_text"].iloc[0]

        random_idx = df.sample(n=1).index[0]
        actual_labels = {}
        for col in TARGET_COLS:
            if col in df.columns:
                actual_labels[col] = bool(df.loc[random_idx, col])

        return {
            "comment": random_comment,
            "actual_labels": actual_labels if actual_labels else None,
        }

    except Exception as e:
        logger.error(f"Error getting random comment: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving example comment")


@app.post("/feedback")
async def toxicity_feedback(request: ToxicityFeedback) -> dict:
    """
    Toxicity feedback endpoint for collecting user corrections
    Args:
        request (ToxicityFeedback): Feedback about prediction accuracy
    Returns:
        dict: Confirmation message
    """
    try:
        feedback_log = {
            "endpoint": "/feedback",
            "request_text": request.request_text[:100] + "..."
            if len(request.request_text) > 100
            else request.request_text,
            "predicted_labels": request.predicted_labels,
            "predicted_probabilities": request.predicted_probabilities,
            "true_labels": request.true_labels,
            "is_prediction_correct": request.is_prediction_correct,
            "user_comments": request.user_comments,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        prediction_logger.info(feedback_log)
        logger.info(f"Feedback received - Correct: {request.is_prediction_correct}")

        return {
            "message": "Feedback received successfully",
            "feedback_id": f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }

    except Exception as e:
        logger.error(f"Error processing toxicity feedback: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error processing toxicity feedback"
        )


@app.get("/stats")
async def model_stats(model=Depends(get_model)) -> dict:
    """
    Get model statistics and information
    Returns:
        dict: Model metadata and statistics
    """
    try:
        # Try to load metadata if available
        try:
            import json
            from pathlib import Path
            from src.core import PROJECT_ROOT, config

            model_path = config["paths"]["model"]
            metadata_path = (
                Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
            )

            if (PROJECT_ROOT / metadata_path).exists():
                with open(PROJECT_ROOT / metadata_path, "r") as f:
                    metadata = json.load(f)

                return {
                    "model_type": "Multi-label Toxicity Classifier",
                    "toxicity_labels": TARGET_COLS,
                    "training_metrics": metadata.get("training_metrics", {}),
                    "model_info": metadata.get("model_info", {}),
                    "dataset_info": metadata.get("dataset_metadata", {}),
                }
        except Exception:
            pass

        # Fallback if metadata not available
        return {
            "model_type": "Multi-label Toxicity Classifier",
            "toxicity_labels": TARGET_COLS,
            "status": "Model loaded successfully",
            "note": "Detailed metrics not available",
        }

    except Exception as e:
        logger.error(f"Error retrieving model stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving model statistics")


@app.post("/moderate")
async def moderate_content(
    request: ModerationRequest, model=Depends(get_model)
) -> ModerationResponse:
    """
    Main moderation endpoint - analyzes content and makes automated moderation decisions

    Args:
        request (ModerationRequest): Content to moderate with optional context

    Returns:
        ModerationResponse: Moderation decision with confidence and reasoning
    """
    try:
        # Get toxicity predictions (reuse existing logic)
        cleaned_text = clean_text(request.text)

        start_time = time.time()
        binary_predictions = model.predict([cleaned_text])[0]
        probability_predictions = model.predict_proba([cleaned_text])
        end_time = time.time()
        latency = end_time - start_time

        # Convert to dict format for moderation logic
        binary_dict = {}
        prob_dict = {}
        for i, label in enumerate(TARGET_COLS):
            binary_dict[label] = bool(binary_predictions[i])
            prob_dict[label] = float(probability_predictions[i][0, 1])

        # Apply moderation rules
        moderation_decision = apply_moderation_rules(
            binary_dict, prob_dict, request.context
        )

        # Queue for human review if needed
        queue_id = None
        if moderation_decision.action == "human_review":
            queue_id = queue_for_review(
                request.text, binary_dict, prob_dict, request.context, request.user_id
            )
            moderation_decision.queue_id = queue_id

        # Log moderation activity
        moderation_log = {
            "endpoint": "/moderate",
            "request_text": request.text[:100] + "..."
            if len(request.text) > 100
            else request.text,
            "context": request.context,
            "user_id": request.user_id,
            "decision": moderation_decision.action,
            "confidence": moderation_decision.confidence,
            "queue_id": queue_id,
            "prediction_latency": latency,
        }
        prediction_logger.info(moderation_log)

        return ModerationResponse(
            decision=moderation_decision.action,
            confidence=moderation_decision.confidence,
            review_required=moderation_decision.action == "human_review",
            queue_id=queue_id,
            reasoning=moderation_decision.reasoning,
        )

    except Exception as e:
        logger.error(f"Error in moderation endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error processing moderation request"
        )


@app.get("/moderation-queue")
async def get_moderation_queue_endpoint(
    status: str = "pending", priority: Optional[str] = None, limit: Optional[int] = 50
) -> dict:
    """
    Get items from the moderation queue for human review

    Args:
        status: Filter by status ("pending", "reviewed", "all")
        priority: Filter by priority ("high", "medium", "low")
        limit: Maximum number of items to return (default 50)

    Returns:
        Dict with queue items and statistics
    """
    try:
        queue_items = get_moderation_queue(status, priority, limit)
        queue_stats = get_queue_stats()

        return {
            "items": queue_items,
            "stats": queue_stats,
            "total_returned": len(queue_items),
        }

    except Exception as e:
        logger.error(f"Error retrieving moderation queue: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving moderation queue")


@app.post("/moderation-queue/{queue_id}/review")
async def review_content(queue_id: str, decision: ReviewDecision) -> dict:
    """
    Process human moderator decision on queued content

    Args:
        queue_id: ID of the queue item to review
        decision: ReviewDecision with action and optional notes

    Returns:
        Dict with confirmation of review processing
    """
    try:
        success = process_review_decision(
            queue_id, decision.action, decision.moderator_notes
        )

        if not success:
            raise HTTPException(status_code=404, detail="Queue item not found")

        # Log review decision
        review_log = {
            "endpoint": "/moderation-queue/review",
            "queue_id": queue_id,
            "action": decision.action,
            "moderator_notes": decision.moderator_notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        prediction_logger.info(review_log)

        return {
            "message": "Review decision processed successfully",
            "queue_id": queue_id,
            "action": decision.action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing review decision: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing review decision")


@app.get("/moderation-stats")
async def get_moderation_stats() -> dict:
    """
    Get moderation queue statistics and metrics

    Returns:
        Dict with comprehensive moderation statistics
    """
    try:
        stats = get_queue_stats()

        return {
            "queue_statistics": stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error retrieving moderation stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Error retrieving moderation statistics"
        )


@app.get("/favicon.ico")
async def favicon():
    """
    Handles constant favicon requests from browser
    to prevent 404 Error logs.
    """
    return Response(status_code=204, content="No favicon here!")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "This is a test comment"}'

curl -X POST "http://localhost:8000/predict_proba" \
-H "Content-Type: application/json" \
-d '{"text": "This is a great comment!"}'

curl -X GET "http://localhost:8000/stats" \
-H "accept: application/json"

curl -X GET "http://localhost:8000/example" \
-H "accept: application/json"
"""
