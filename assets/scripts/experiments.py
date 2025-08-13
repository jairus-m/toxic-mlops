import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import re
import warnings

# Try to import XGBoost, but continue if not available
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"XGBoost not available: {e}")
    print(
        "Continuing without XGBoost. Only Logistic Regression and Random Forest will be used."
    )
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# Set up MLflow experiment
mlflow.set_experiment("toxic-comments-classification")


def train_and_evaluate_model(
    model_name, classifier, X_train, X_test, y_train, y_test, target_cols
):
    """
    Train and evaluate a model while logging metrics to MLflow
    """
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        params = classifier.get_params()
        mlflow.log_params(params)

        # Create and train pipeline
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            lowercase=True,
            strip_accents="ascii",
        )

        multi_clf = MultiOutputClassifier(classifier, n_jobs=-1)
        pipeline = Pipeline([("tfidf", tfidf), ("classifier", multi_clf)])

        # Train model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        y_pred_proba_array = np.column_stack([prob[:, 1] for prob in y_pred_proba])

        # Calculate metrics
        exact_match_accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("exact_match_accuracy", exact_match_accuracy)

        # Per-label metrics
        mean_auc = 0
        for i, col in enumerate(target_cols):
            # Accuracy
            acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
            mlflow.log_metric(f"{col}_accuracy", acc)

            # AUC
            try:
                auc = roc_auc_score(y_test.iloc[:, i], y_pred_proba_array[:, i])
                mlflow.log_metric(f"{col}_auc", auc)
                mean_auc += auc
            except ValueError as e:
                print(f"{col}: Error calculating AUC - {e}")

        mean_auc /= len(target_cols)
        mlflow.log_metric("mean_auc", mean_auc)

        # Create model signature
        from mlflow.types.schema import Schema, ColSpec

        # Input schema: single string column for text
        input_schema = Schema([ColSpec("string", "text")])

        # Output schema: multiple binary columns for each toxicity type
        output_schema = Schema([ColSpec("integer", col) for col in target_cols])

        # Create signature
        from mlflow.models.signature import ModelSignature

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Create input example
        input_example = {"text": "This is an example comment for model signature"}

        # Log the model with signature and input example
        mlflow.sklearn.log_model(
            pipeline, "model", signature=signature, input_example=input_example
        )

        print(f"\n{model_name} Results:")
        print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
        print(f"Mean ROC-AUC: {mean_auc:.4f}")

        return pipeline, exact_match_accuracy, mean_auc


# Load and preprocess data
print("Loading data...")
train_df = pd.read_csv("assets/data/train.csv")

# Define target columns
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# Clean the text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())
    return text


print("Cleaning text...")
train_df["comment_text_clean"] = train_df["comment_text"].apply(clean_text)

# Split the data
print("Splitting data...")
X = train_df["comment_text_clean"]
y = train_df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["toxic"]
)

# Train different models
print("\nTraining models...")

# Dictionary to store model results
model_results = {}

# 1. Logistic Regression
lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
)
lr_pipeline, lr_acc, lr_auc = train_and_evaluate_model(
    "LogisticRegression", lr, X_train, X_test, y_train, y_test, target_cols
)
model_results["LogisticRegression"] = {
    "pipeline": lr_pipeline,
    "acc": lr_acc,
    "auc": lr_auc,
}

# 2. Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
)
rf_pipeline, rf_acc, rf_auc = train_and_evaluate_model(
    "RandomForest", rf, X_train, X_test, y_train, y_test, target_cols
)
model_results["RandomForest"] = {"pipeline": rf_pipeline, "acc": rf_acc, "auc": rf_auc}

# 3. XGBoost (if available)
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    xgb_pipeline, xgb_acc, xgb_auc = train_and_evaluate_model(
        "XGBoost", xgb, X_train, X_test, y_train, y_test, target_cols
    )
    model_results["XGBoost"] = {
        "pipeline": xgb_pipeline,
        "acc": xgb_acc,
        "auc": xgb_auc,
    }

# Print final comparison
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
for model_name, results in model_results.items():
    print(f"{model_name} - Accuracy: {results['acc']:.4f}, AUC: {results['auc']:.4f}")

# Select best model based on AUC
best_model_name = max(model_results.items(), key=lambda x: x[1]["auc"])[0]
best_model_results = model_results[best_model_name]
best_model = best_model_results["pipeline"]
best_auc = best_model_results["auc"]

print(f"\nBest model: {best_model_name} (AUC: {best_auc:.4f})")


# Function to predict with best model
def predict_toxicity(text, pipeline=best_model, target_cols=target_cols):
    """
    Predict toxicity for a new comment using the best model
    """
    cleaned_text = clean_text(text)
    prediction_proba = pipeline.predict_proba([cleaned_text])

    # Extract probabilities
    proba_array = np.array([prob[0, 1] for prob in prediction_proba])

    result = {}
    for i, col in enumerate(target_cols):
        result[col] = {
            "probability": proba_array[i],
            "prediction": "Yes" if proba_array[i] > 0.5 else "No",
        }

    return result


# Test the best model
print("\n" + "=" * 50)
print("TESTING BEST MODEL")
print("=" * 50)

test_comments = [
    "This is a great article, thank you for sharing!",
    "You are an idiot and should shut up!",
    "I disagree with your opinion but respect your right to express it.",
]

for i, comment in enumerate(test_comments):
    print(f"\nTest Comment {i + 1}: {comment}")
    predictions = predict_toxicity(comment)
    for label, result in predictions.items():
        print(f"  {label}: {result['probability']:.3f} ({result['prediction']})")
