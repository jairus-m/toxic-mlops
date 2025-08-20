from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert json_response["model_ready"] is True
    assert json_response["toxicity_labels"] == TARGET_COLS
    assert (
        "timestamp" in json_response
    )  # Check timestamp exists but don't compare value


def test_predict(client, mock_prediction_logger):
    """Test prediction endpoint"""
    response = client.post("/predict", json={"text": "This is an insult."})
    assert response.status_code == 200
    expected_response = {
        "is_toxic": True,
        "predictions": {
            "toxic": True,
            "insult": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "identity_hate": False,
        },
    }
    assert response.json() == expected_response
    mock_prediction_logger.info.assert_called_once()


def test_predict_proba(client, mock_prediction_logger):
    """Test probability prediction endpoint"""
    response = client.post("/predict_proba", json={"text": "This is an insult."})
    assert response.status_code == 200
    expected_response = {
        "is_toxic": True,
        "max_toxicity_probability": 0.8,
        "predictions": {
            "toxic": True,
            "insult": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "identity_hate": False,
        },
        "probabilities": {
            "toxic": 0.8,
            "severe_toxic": 0.1,
            "obscene": 0.05,
            "threat": 0.02,
            "insult": 0.7,
            "identity_hate": 0.01,
        },
    }
    assert response.json() == expected_response
    mock_prediction_logger.info.assert_called_once()


def test_example(client):
    """Test example endpoint"""
    mock_df = pd.DataFrame(
        {
            "comment_text": ["A random non-toxic comment."],
            "toxic": [0],
            "severe_toxic": [0],
            "obscene": [0],
            "threat": [0],
            "insult": [0],
            "identity_hate": [0],
        }
    )
    with (
        patch("src.fastapi_backend.main.pd.read_csv", return_value=mock_df),
        patch(
            "src.fastapi_backend.main.get_asset_path",
            return_value="/mock/path/examples.csv",
        ),
    ):
        response = client.get("/example")
        assert response.status_code == 200
        expected_response = {
            "comment": "A random non-toxic comment.",
            "actual_labels": {
                "toxic": False,
                "severe_toxic": False,
                "obscene": False,
                "threat": False,
                "insult": False,
                "identity_hate": False,
            },
        }
        assert response.json() == expected_response


def test_feedback(client, mock_prediction_logger):
    """Test feedback endpoint"""
    feedback_data = {
        "request_text": "You are such an idiot!",
        "predicted_labels": {
            "toxic": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "insult": True,
            "identity_hate": False,
        },
        "predicted_probabilities": {
            "toxic": 0.8234,
            "severe_toxic": 0.1245,
            "obscene": 0.0892,
            "threat": 0.0156,
            "insult": 0.7123,
            "identity_hate": 0.0234,
        },
        "true_labels": {
            "toxic": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "insult": True,
            "identity_hate": False,
        },
        "is_prediction_correct": True,
        "user_comments": "The prediction was accurate.",
    }
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert "message" in response.json()
    assert "feedback_id" in response.json()
    mock_prediction_logger.info.assert_called_once()


def test_stats(client):
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    json_response = response.json()
    assert "model_type" in json_response
    assert "toxicity_labels" in json_response
    assert json_response["toxicity_labels"] == TARGET_COLS


def test_moderate_approve(client, mock_prediction_logger):
    """Test moderation endpoint with a non-toxic comment"""
    response = client.post("/moderate", json={"text": "This is a friendly comment."})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["decision"] == "approve"
    assert json_response["review_required"] is False
    mock_prediction_logger.info.assert_called_once()


def test_moderate_human_review(client, mock_prediction_logger):
    """Test moderation endpoint with a comment that needs human review"""
    response = client.post("/moderate", json={"text": "This is an insult."})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["decision"] == "human_review"
    assert json_response["review_required"] is True
    mock_prediction_logger.info.assert_called_once()


def test_moderate_reject(client, app, mock_prediction_logger):
    """Test moderation endpoint with a highly toxic comment"""
    # To trigger the reject, we need to override the model dependency for this specific test
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[1, 1, 1, 0, 1, 0]])
    mock_model.predict_proba.return_value = [
        np.array([[0.05, 0.95]]),  # toxic
        np.array([[0.1, 0.9]]),  # severe_toxic
        np.array([[0.05, 0.95]]),  # obscene
        np.array([[0.98, 0.02]]),  # threat
        np.array([[0.1, 0.9]]),  # insult
        np.array([[0.99, 0.01]]),  # identity_hate
    ]

    from src.fastapi_backend.main import get_model

    app.dependency_overrides[get_model] = lambda: mock_model

    response = client.post("/moderate", json={"text": "This is extremely toxic."})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["decision"] == "reject"
    assert json_response["review_required"] is False
    mock_prediction_logger.info.assert_called_once()

    # Clean up the override
    app.dependency_overrides = {}


def test_get_moderation_queue(client):
    """Test get moderation queue endpoint"""
    with (
        patch(
            "src.fastapi_backend.main.get_moderation_queue",
            return_value=[{"id": "1", "text": "Test"}],
        ),
        patch(
            "src.fastapi_backend.main.get_queue_stats",
            return_value={"pending": 1, "reviewed": 0},
        ),
    ):
        response = client.get("/moderation-queue")
        assert response.status_code == 200
        json_response = response.json()
        assert "items" in json_response
        assert "stats" in json_response
        assert len(json_response["items"]) == 1


def test_review_content(client, mock_prediction_logger):
    """Test review content endpoint"""
    with patch(
        "src.fastapi_backend.main.process_review_decision", return_value=True
    ) as mock_process:
        response = client.post(
            "/moderation-queue/123/review",
            json={"action": "approve", "moderator_notes": "Looks fine."},
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["message"] == "Review decision processed successfully"
        mock_process.assert_called_once_with("123", "approve", "Looks fine.")
        mock_prediction_logger.info.assert_called_once()


def test_get_moderation_stats(client):
    """Test get moderation stats endpoint"""
    with patch(
        "src.fastapi_backend.main.get_queue_stats",
        return_value={"pending": 5, "reviewed": 10, "approved": 8, "rejected": 2},
    ) as mock_stats:
        response = client.get("/moderation-stats")
        assert response.status_code == 200
        json_response = response.json()
        assert "queue_statistics" in json_response
        assert json_response["queue_statistics"]["pending"] == 5
        mock_stats.assert_called_once()
