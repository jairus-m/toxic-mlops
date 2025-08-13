from unittest.mock import patch
import pandas as pd

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert json_response["model_ready"] is True
    assert json_response["toxicity_labels"] == TARGET_COLS


def test_predict(client, mock_prediction_logger):
    """Test prediction endpoint"""
    response = client.post("/predict", json={"text": "This is an insult."})
    assert response.status_code == 200
    json_response = response.json()
    assert "is_toxic" in json_response
    assert "predictions" in json_response
    assert list(json_response["predictions"].keys()) == TARGET_COLS
    # Based on mock_model, toxic and insult should be true
    assert json_response["predictions"]["toxic"] is True
    assert json_response["predictions"]["insult"] is True
    assert json_response["predictions"]["obscene"] is False
    mock_prediction_logger.info.assert_called_once()


def test_predict_proba(client, mock_prediction_logger):
    """Test probability prediction endpoint"""
    response = client.post("/predict_proba", json={"text": "This is an insult."})
    assert response.status_code == 200
    json_response = response.json()
    assert "is_toxic" in json_response
    assert "max_toxicity_probability" in json_response
    assert "predictions" in json_response
    assert "probabilities" in json_response
    assert list(json_response["probabilities"].keys()) == TARGET_COLS
    # Based on mock_model, check a few probabilities
    assert json_response["probabilities"]["toxic"] == 0.8
    assert json_response["probabilities"]["insult"] == 0.7
    assert json_response["max_toxicity_probability"] == 0.8
    mock_prediction_logger.info.assert_called_once()


def test_example(client):
    """Test example endpoint"""
    # Mock the dataframe that would be loaded from the CSV
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
    with patch("src.fastapi_backend.main.pd.read_csv", return_value=mock_df):
        response = client.get("/example")
        assert response.status_code == 200
        json_response = response.json()
        assert "comment" in json_response
        assert "actual_labels" in json_response
        assert json_response["comment"] == "A random non-toxic comment."
        assert json_response["actual_labels"]["toxic"] is False


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
