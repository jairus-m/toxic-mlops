"""
Pydantic models for the Toxic Comment Classification FastAPI app
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional


class PredictRequest(BaseModel):
    """Request Pydantic model for toxicity prediction sent to the API
    as a JSON object with a single key "text".

    Example request:
        {
            "text": "This is a sample comment to analyze for toxicity."
        }
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Comment text to analyze for toxicity",
    )


class ToxicityResponse(BaseModel):
    """Response model for toxicity prediction returned by the API
    with binary predictions for each toxicity type.

    Example response:
        {
            "is_toxic": true,
            "predictions": {
                "toxic": true,
                "severe_toxic": false,
                "obscene": false,
                "threat": false,
                "insult": true,
                "identity_hate": false
            }
        }
    """

    is_toxic: bool = Field(..., description="Overall toxicity indicator")
    predictions: Dict[str, bool] = Field(
        ..., description="Binary predictions for each toxicity type"
    )


class ToxicityProbabilityResponse(BaseModel):
    """Response model for toxicity prediction with probability scores returned by the API.

    Example response:
        {
            "is_toxic": true,
            "max_toxicity_probability": 0.8234,
            "predictions": {
                "toxic": true,
                "severe_toxic": false,
                "obscene": false,
                "threat": false,
                "insult": true,
                "identity_hate": false
            },
            "probabilities": {
                "toxic": 0.8234,
                "severe_toxic": 0.1245,
                "obscene": 0.0892,
                "threat": 0.0156,
                "insult": 0.7123,
                "identity_hate": 0.0234
            }
        }
    """

    is_toxic: bool = Field(..., description="Overall toxicity indicator")
    max_toxicity_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Highest probability among all toxicity types"
    )
    predictions: Dict[str, bool] = Field(
        ..., description="Binary predictions for each toxicity type"
    )
    probabilities: Dict[str, float] = Field(
        ..., description="Probability scores for each toxicity type"
    )


class ExampleResponse(BaseModel):
    """Response model for example comment returned by the API.

    Example response:
        {
            "comment": "This movie was absolutely terrible and a waste of time.",
            "actual_labels": {
                "toxic": false,
                "severe_toxic": false,
                "obscene": false,
                "threat": false,
                "insult": false,
                "identity_hate": false
            }
        }
    """

    comment: str = Field(..., description="Example comment from the dataset")
    actual_labels: Optional[Dict[str, bool]] = Field(
        None, description="Actual labels from the dataset if available"
    )


class ToxicityFeedback(BaseModel):
    """Request model for toxicity prediction feedback sent to the API.

    Example request:
        {
            "request_text": "You are such an idiot!",
            "predicted_labels": {
                "toxic": true,
                "severe_toxic": false,
                "obscene": false,
                "threat": false,
                "insult": true,
                "identity_hate": false
            },
            "predicted_probabilities": {
                "toxic": 0.8234,
                "severe_toxic": 0.1245,
                "obscene": 0.0892,
                "threat": 0.0156,
                "insult": 0.7123,
                "identity_hate": 0.0234
            },
            "true_labels": {
                "toxic": true,
                "severe_toxic": false,
                "obscene": false,
                "threat": false,
                "insult": true,
                "identity_hate": false
            },
            "is_prediction_correct": true,
            "user_comments": "The prediction was accurate."
        }
    """

    request_text: str = Field(
        ..., description="Original comment text that was analyzed"
    )
    predicted_labels: Dict[str, bool] = Field(
        ..., description="Labels predicted by the model"
    )
    predicted_probabilities: Dict[str, float] = Field(
        ..., description="Probabilities predicted by the model"
    )
    true_labels: Dict[str, bool] = Field(
        ..., description="Correct labels as determined by the user"
    )
    is_prediction_correct: bool = Field(
        ..., description="Whether the prediction was correct overall"
    )
    user_comments: Optional[str] = Field(
        None, description="Additional feedback comments from the user"
    )


# Legacy schemas for backward compatibility (if needed)
class SentimentResponse(BaseModel):
    """Legacy response model for sentiment prediction - kept for backward compatibility"""

    sentiment: str


class SentimentProbabilityResponse(BaseModel):
    """Legacy response model for sentiment prediction with probability - kept for backward compatibility"""

    sentiment: str
    probability: float


class SentimentFeedback(BaseModel):
    """Legacy feedback model - kept for backward compatibility"""

    request_text: str
    predicted_sentiment: str
    probability: float
    true_sentiment: str
    is_sentiment_correct: bool
