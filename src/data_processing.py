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
    "very", "really", "so", "too", "that", "much", "quite",
    "extremely", "super", "totally", "absolutely", "really"
}

SKIP_WORDS = {
    "feel", "feels", "feeling", "felt",
    "look", "looks", "looking",
    "seem", "seems", "seeming",
    "sound", "sounds", "sounding",
    "be", "being",
    "become", "becomes", "becoming",
    "get", "gets", "getting",
    "make", "makes", "making",
    "cause", "causes", "causing",
    "create", "creates", "creating"
}

STOP_CONNECTORS = {
    "but", "however", "although", "though",
    "because", "while", "and", "or", "then",
    "if", "when", "after", "before"
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "happy", "safe", "hope", "helpful",
    "useful", "better", "best", "love", "like", "positive", "beneficial",
    "effective", "calm", "protected", "prepared", "ready", "recovering",
    "improving", "stable", "successful", "encouraging", "relieved",
    "normal", "controlled", "decreasing", "dropping", "lower"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "sad", "angry", "scared", "afraid", "worried",
    "anxious", "stressed", "panic", "fear", "worse", "worst", "hate",
    "awful", "horrible", "ruined", "destroyed", "sick", "dangerous",
    "risky", "risk", "death", "dead", "unsafe", "severe", "spreading",
    "increasing", "rising", "suffering", "full", "pressure", "failed",
    "failing", "shortage", "shortages", "crowded", "infected", "infection",
    "outbreak", "wave"
}

# Words li ila jaw m3a negation kaywelliw positive
NEGATION_TO_POS = {
    "bad", "terrible", "awful", "horrible",
    "scared", "afraid", "worried", "anxious", "stressed",
    "dangerous", "risky", "severe", "panic", "fear",
    "high", "increasing", "rising", "spreading",
    "worse", "worst", "sick", "infected"
}

# Words li ila jaw m3a negation kaywelliw negative
NEGATION_TO_NEG = {
    "good", "great", "happy", "safe", "helpful", "useful",
    "better", "effective", "protected", "prepared", "ready",
    "working", "recovering", "improving", "calm", "stable",
    "controlled", "successful"
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
    i = 0

    while i < len(words):
        word = words[i]
        new_words.append(word)

        if word in NEGATION_WORDS:
            window = words[i + 1:i + 7]

            # no longer scared -> no_longer_scared_pos
            if len(window) >= 2 and window[0] == "longer":
                target = window[1]

                if target in NEGATION_TO_POS or target in NEGATIVE_WORDS:
                    new_words.append(f"{word}_longer_{target}_pos")
                    i += 1
                    continue

                if target in NEGATION_TO_NEG or target in POSITIVE_WORDS:
                    new_words.append(f"{word}_longer_{target}_neg")
                    i += 1
                    continue

            # not under control -> not_under_control_neg
            if len(window) >= 2 and window[0] == "under" and window[1] == "control":
                new_words.append(f"{word}_under_control_neg")
                i += 1
                continue

            for candidate in window:
                if candidate in STOP_CONNECTORS:
                    break

                if candidate in MODIFIERS or candidate in SKIP_WORDS:
                    continue

                # not scared / not dangerous / no panic => positive
                if candidate in NEGATION_TO_POS:
                    new_words.append(f"{word}_{candidate}_pos")
                    break

                # not safe / not good / not working => negative
                if candidate in NEGATION_TO_NEG:
                    new_words.append(f"{word}_{candidate}_neg")
                    break

                # general rule: not + negative word => positive
                if candidate in NEGATIVE_WORDS:
                    new_words.append(f"{word}_{candidate}_pos")
                    break

                # general rule: not + positive word => negative
                if candidate in POSITIVE_WORDS:
                    new_words.append(f"{word}_{candidate}_neg")
                    break

        i += 1

    return " ".join(new_words)

def prepare_text_for_model(text):
    return handle_negation(text)

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



def normalize_sentiment_value(value):
    value = str(value).lower().strip()

    positive_values = [
        "positive",
        "pos",
        "positif",
        "1",
        "p"
    ]

    negative_values = [
        "negative",
        "neg",
        "negatif",
        "-1",
        "n"
    ]

    neutral_values = [
        "neutral",
        "natural",
        "neu",
        "nat",
        "neutre",
        "0"
    ]

    if value in positive_values:
        return "pos"

    if value in negative_values:
        return "neg"

    if value in neutral_values:
        return "neu"

    return value











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
    tokens = prepared.split()

    current_conf = 0.0 if confidence is None else float(confidence)

    # 1) Negation tokens generated by handle_negation
    positive_negation = any(token.endswith("_pos") for token in tokens)
    negative_negation = any(token.endswith("_neg") for token in tokens)

    if positive_negation:
        return "pos", max(current_conf, 0.92), "positive negation rule"

    if negative_negation:
        return "neg", max(current_conf, 0.92), "negative negation rule"

    # 2) Clear negative phrases only, not single words like scared/fear
    clear_negative_phrases = [
        "i am scared",
        "people are scared",
        "families are scared",
        "students are scared",
        "i am afraid",
        "people are afraid",
        "i am worried",
        "people are worried",
        "families are worried",
        "i feel unsafe",
        "people feel unsafe",
        "workers feel unsafe",
        "the situation is dangerous",
        "the virus is dangerous",
        "the variant is dangerous",
        "cases are increasing",
        "cases are rising",
        "infections are increasing",
        "the virus is spreading",
        "there is panic",
        "hospital is full",
        "hospitals are full",
        "patients are suffering",
        "death rate is increasing",
        "health system is under pressure"
    ]

    clear_positive_phrases = [
        "people feel safe",
        "people are safe",
        "families feel safe",
        "the situation is better",
        "cases are decreasing",
        "cases are dropping",
        "risk is lower",
        "under control",
        "vaccine is effective",
        "patients are recovering",
        "patients are improving",
        "good news",
        "positive update",
        "hospital is calm",
        "people are calm"
    ]

    has_clear_negative = any(phrase in clean for phrase in clear_negative_phrases)
    has_clear_positive = any(phrase in clean for phrase in clear_positive_phrases)

    if has_clear_negative:
        return "neg", max(current_conf, 0.85), "clear negative rule"

    if has_clear_positive:
        return "pos", max(current_conf, 0.85), "clear positive rule"

    return predicted_label, confidence, None

