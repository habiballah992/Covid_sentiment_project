from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "Covid_datasets.csv"


CONTRACTIONS = {
    "i'm": "i am",
    "im": "i am",
    "you're": "you are",
    "youre": "you are",
    "it's": "it is",
    "its": "it is",
    "can't": "can not",
    "cant": "can not",
    "won't": "will not",
    "wont": "will not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "haven't": "have not",
    "havent": "have not",
    "hasn't": "has not",
    "hasnt": "has not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "couldn't": "could not",
    "couldnt": "could not",
}


NEGATION_WORDS = {
    "not", "no", "never", "cannot", "without"
}

MODIFIERS = {
    "very", "really", "so", "too", "that", "much", "quite", "extremely",
    "super", "totally", "absolutely"
}

SENTIMENT_WORDS = {
    "good", "great", "excellent", "happy", "safe", "hope", "helpful",
    "useful", "better", "best", "love", "like", "positive", "beneficial",
    "bad", "terrible", "sad", "angry", "scared", "worried", "anxious",
    "fear", "worse", "worst", "hate", "awful", "horrible", "ruined",
    "destroyed", "sick", "dangerous", "risk", "death", "dead"
}


def expand_contractions(text):
    text = str(text).lower()

    for short, full in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", full, text)

    text = re.sub(r"n['’]t\b", " not", text)

    return text


def basic_clean_text(text):
    text = expand_contractions(text)

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-zA-Z_ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def handle_negation(text):
    text = basic_clean_text(text)
    words = text.split()

    new_words = []

    for i, word in enumerate(words):
        new_words.append(word)

        if word in NEGATION_WORDS:
            window = words[i + 1:i + 5]

            for candidate in window:
                if candidate in MODIFIERS:
                    continue

                if candidate in SENTIMENT_WORDS:
                    new_words.append(f"not_{candidate}")
                    break

    return " ".join(new_words)


def prepare_text_for_model(text):
    return handle_negation(text)


def normalize_sentiment_value(value):
    value = str(value).lower().strip()

    if value in ["positive", "pos", "1"]:
        return "pos"

    if value in ["negative", "neg", "-1"]:
        return "neg"

    if value in ["neutral", "natural", "neu", "nat", "0"]:
        return "neu"

    return value


def detect_columns(df):
    text_candidates = [
        "clean_tweet",
        "tweet",
        "text",
        "comment",
        "content",
        "sentence",
        "message",
        "review"
    ]

    label_candidates = [
        "sentiment",
        "label",
        "target",
        "class",
        "category"
    ]

    lower_map = {col.lower().strip(): col for col in df.columns}

    text_col = None
    label_col = None

    for col in text_candidates:
        if col in lower_map:
            text_col = lower_map[col]
            break

    for col in label_candidates:
        if col in lower_map:
            label_col = lower_map[col]
            break

    return text_col, label_col


def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, on_bad_lines="skip")

    candidates = sorted(DATA_DIR.rglob("*.csv"))

    if candidates:
        print(f"Dataset found automatically: {candidates[0]}")
        return pd.read_csv(candidates[0], on_bad_lines="skip")

    raise FileNotFoundError(
        f"Dataset not found. Put your CSV here: {DATA_PATH}"
    )


def cleaning_data(df):
    text_col, label_col = detect_columns(df)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect text/sentiment columns. Found columns: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].copy()
    df.columns = ["clean_tweet", "sentiment"]

    df = df.dropna(subset=["clean_tweet", "sentiment"])

    df["clean_tweet"] = df["clean_tweet"].astype(str).apply(prepare_text_for_model)
    df["sentiment"] = df["sentiment"].apply(normalize_sentiment_value)

    df = df[df["clean_tweet"].str.strip() != ""]
    df = df[df["sentiment"].isin(["pos", "neg", "neu"])]

    df = df.drop_duplicates(subset=["clean_tweet"])

    print("\nDataset shape after cleaning:", df.shape)
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    return df


def split_data(df):
    x = df["clean_tweet"]
    y = df["sentiment"]

    stratify_value = y if y.value_counts().min() >= 2 else None

    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )


def vectorize_data(x_train, x_test):
    MODELS_DIR.mkdir(exist_ok=True)

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z_]{2,}\b",
        dtype=np.float32
    )

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")

    print("\nVectorizer saved successfully.")
    print("Train matrix shape:", x_train_vectorized.shape)
    print("Test matrix shape:", x_test_vectorized.shape)

    return x_train_vectorized, x_test_vectorized, vectorizer


def apply_sentiment_rules(original_text, predicted_label, confidence=None):
    clean = basic_clean_text(original_text)
    prepared = prepare_text_for_model(original_text)

    negative_phrases = [
        "not happy",
        "not good",
        "not safe",
        "not satisfied",
        "not useful",
        "not helpful",
        "not better",
        "not comfortable",
        "not okay",
        "i hate",
        "hate covid",
        "bad",
        "terrible",
        "awful",
        "horrible",
        "sad",
        "angry",
        "scared",
        "worried",
        "anxious",
        "fear",
        "worse",
        "worst",
        "ruined",
        "destroyed",
        "dangerous"
    ]

    positive_phrases = [
        "not bad",
        "good",
        "great",
        "excellent",
        "happy",
        "safe",
        "hope",
        "helpful",
        "useful",
        "better",
        "best",
        "love",
        "positive",
        "beneficial",
        "effective"
    ]

    negative_tokens = [
        "not_happy",
        "not_good",
        "not_safe",
        "not_satisfied",
        "not_useful",
        "not_helpful",
        "not_better"
    ]

    current_conf = 0.0 if confidence is None else float(confidence)

    has_negative = any(phrase in clean for phrase in negative_phrases)
    has_negative = has_negative or any(token in prepared for token in negative_tokens)

    has_positive = any(phrase in clean for phrase in positive_phrases)

    if clean.startswith("not bad"):
        return "pos", max(current_conf, 0.82), "positive phrase rule: not bad"

    if has_negative:
        if normalize_sentiment_value(predicted_label) != "neg" or current_conf < 0.90:
            return "neg", max(current_conf, 0.90), "negative phrase / negation rule"

    if has_positive and not has_negative:
        if normalize_sentiment_value(predicted_label) != "pos" or current_conf < 0.85:
            return "pos", max(current_conf, 0.85), "positive phrase rule"

    return predicted_label, confidence, None