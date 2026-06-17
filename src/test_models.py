from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_processing import prepare_text_for_model, apply_sentiment_rules


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def load_model_vectorizer(model_name="svm"):
    model = joblib.load(MODELS_DIR / f"{model_name}.pkl")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")

    return model, vectorizer


def softmax(scores):
    scores = np.array(scores, dtype=float)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)

    return exp_scores / exp_scores.sum()


def predict_sentiment(text, model_name="svm", use_rules=True):
    model, vectorizer = load_model_vectorizer(model_name)

    cleaned_text = prepare_text_for_model(text)
    x = vectorizer.transform([cleaned_text])

    prediction = model.predict(x)[0]
    confidence = None

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        scores = np.array(scores)

        if scores.ndim == 1:
            scores = scores.ravel()
        else:
            scores = scores[0]

        probabilities = softmax(scores)
        confidence = float(np.max(probabilities))

    if use_rules:
        prediction, confidence, rule = apply_sentiment_rules(
            text,
            prediction,
            confidence
        )

    return prediction


def test_model(model, x_test):
    return model.predict(x_test)


def evaluate_model(y_test, y_pred):
    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))