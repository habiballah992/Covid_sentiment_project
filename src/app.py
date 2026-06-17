import re
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from data_processing import (
    prepare_text_for_model,
    apply_sentiment_rules,
    normalize_sentiment_value,
    detect_columns
)

from audio_processing import load_whisper_model, audio_file_to_text


DATA_DIR = ROOT_DIR / "data"
MAIN_DATASET_PATH = DATA_DIR / "Covid_datasets.csv"
MODELS_DIR = ROOT_DIR / "models"


st.set_page_config(
    page_title="Covid Sentiment AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .hero {
            padding: 32px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.96), rgba(15, 23, 42, 0.96));
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 18px 50px rgba(0,0,0,0.25);
            margin-bottom: 25px;
        }

        .hero-title {
            font-size: 52px;
            font-weight: 900;
            line-height: 1.05;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #cbd5e1;
            max-width: 960px;
            line-height: 1.6;
        }

        .card {
            padding: 22px;
            border-radius: 22px;
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.15);
            box-shadow: 0 14px 32px rgba(0,0,0,0.20);
            height: 100%;
        }

        .metric-card {
            padding: 22px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.75));
            border: 1px solid rgba(148, 163, 184, 0.18);
            height: 150px;
        }

        .metric-title {
            font-size: 14px;
            color: #94a3b8;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }

        .metric-value {
            font-size: 34px;
            font-weight: 900;
            margin-top: 8px;
        }

        .metric-subtitle {
            font-size: 13px;
            color: #94a3b8;
            margin-top: 8px;
        }

        .status-ok {
            color: #22c55e;
            font-weight: 800;
        }

        .status-warn {
            color: #f97316;
            font-weight: 800;
        }

        .status-bad {
            color: #ef4444;
            font-weight: 800;
        }

        .result-box {
            padding: 26px;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.22), rgba(147, 51, 234, 0.18));
            border: 1px solid rgba(191, 219, 254, 0.18);
            margin-top: 20px;
        }

        .result-label {
            font-size: 42px;
            font-weight: 900;
            margin-top: 8px;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 14px;
        }

        .pipeline-step {
            padding: 18px;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.14);
            min-height: 160px;
        }

        .pipeline-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }

        .pipeline-title {
            font-size: 18px;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .pipeline-text {
            font-size: 14px;
            color: #cbd5e1;
            line-height: 1.55;
        }

        .stButton > button {
            border-radius: 14px;
            padding: 12px 20px;
            font-weight: 800;
            border: none;
        }

        div[data-testid="stTabs"] button {
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def metric_card(title, value, subtitle="", status_class=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value {status_class}">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def normalize_label(label):
    label = normalize_sentiment_value(label)

    if label == "pos":
        return "positive"

    if label == "neg":
        return "negative"

    if label == "neu":
        return "natural"

    return str(label).lower()


def sentiment_emoji(label):
    label = normalize_label(label)

    if label == "positive":
        return "🟢"

    if label == "negative":
        return "🔴"

    if label == "natural":
        return "🟡"

    return "🔵"


def sentiment_text(label):
    label = normalize_label(label)

    if label == "positive":
        return "Positive"

    if label == "negative":
        return "Negative"

    if label == "natural":
        return "Natural"

    return str(label).capitalize()


def softmax(scores):
    scores = np.array(scores, dtype=float)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)

    return exp_scores / exp_scores.sum()


def get_available_models():
    if not MODELS_DIR.exists():
        return []

    models = []

    for file in MODELS_DIR.glob("*.pkl"):
        if file.name != "vectorizer.pkl":
            models.append(file.stem)

    return sorted(models)


def get_dataset_candidates():
    if not DATA_DIR.exists():
        return []

    return sorted(DATA_DIR.rglob("*.csv"))


@st.cache_data
def load_dataset():
    if MAIN_DATASET_PATH.exists():
        return pd.read_csv(MAIN_DATASET_PATH, on_bad_lines="skip"), str(MAIN_DATASET_PATH)

    candidates = get_dataset_candidates()

    if candidates:
        return pd.read_csv(candidates[0], on_bad_lines="skip"), str(candidates[0])

    return None, None


@st.cache_resource
def load_sentiment_assets(model_name):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


@st.cache_resource
def get_whisper_model(model_size):
    return load_whisper_model(model_size)


def make_rule_probabilities(classes, final_label, confidence):
    if classes is None or len(classes) == 0:
        classes = ["neg", "neu", "pos"]

    classes = list(classes)
    confidence = float(confidence)

    remaining = max(1.0 - confidence, 0.0)
    other_count = max(len(classes) - 1, 1)

    scores = []

    for cls in classes:
        if normalize_label(cls) == normalize_label(final_label):
            scores.append(confidence)
        else:
            scores.append(remaining / other_count)

    return pd.DataFrame({
        "sentiment": classes,
        "score": scores
    })


def demo_predict(text):
    cleaned_text = prepare_text_for_model(text)
    final_label, confidence, rule = apply_sentiment_rules(text, "neu", 0.55)

    if rule is None:
        final_label = "neu"
        confidence = 0.55

    probabilities = make_rule_probabilities(
        ["neg", "neu", "pos"],
        final_label,
        confidence
    )

    return {
        "label": final_label,
        "confidence": confidence,
        "cleaned_text": cleaned_text,
        "probabilities": probabilities,
        "engine": "Demo mode",
        "rule": rule
    }


def predict_sentiment_ui(text, model=None, vectorizer=None):
    cleaned_text = prepare_text_for_model(text)

    if model is None or vectorizer is None:
        return demo_predict(text)

    x = vectorizer.transform([cleaned_text])
    prediction = model.predict(x)[0]

    classes = getattr(model, "classes_", None)
    confidence = None
    probabilities_df = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]
        confidence = float(np.max(probabilities))

        probabilities_df = pd.DataFrame({
            "sentiment": classes,
            "score": probabilities
        })

    elif hasattr(model, "decision_function") and classes is not None:
        scores = model.decision_function(x)
        scores = np.array(scores)

        if scores.ndim == 1:
            scores = scores.ravel()
        else:
            scores = scores[0]

        probabilities = softmax(scores)
        confidence = float(np.max(probabilities))

        probabilities_df = pd.DataFrame({
            "sentiment": classes,
            "score": probabilities
        })

    final_label, final_confidence, rule = apply_sentiment_rules(
        text,
        prediction,
        confidence
    )

    engine = "Trained model"

    if rule is not None:
        engine = "Trained model + smart correction"
        confidence = final_confidence
        probabilities_df = make_rule_probabilities(classes, final_label, confidence)
        prediction = final_label

    return {
        "label": prediction,
        "confidence": confidence,
        "cleaned_text": cleaned_text,
        "probabilities": probabilities_df,
        "engine": engine,
        "rule": rule
    }


def save_prediction_history(source, original_text, result):
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.insert(0, {
        "source": source,
        "text": str(original_text)[:160],
        "sentiment": sentiment_text(result["label"]),
        "confidence": None if result["confidence"] is None else round(result["confidence"] * 100, 2),
        "engine": result["engine"]
    })

    st.session_state.history = st.session_state.history[:30]


def show_prediction_result(result):
    label = result["label"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    confidence_text = "Not available"

    if confidence is not None:
        confidence_text = f"{confidence * 100:.2f}%"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="small-muted">Detected sentiment</div>
            <div class="result-label">{sentiment_emoji(label)} {sentiment_text(label).upper()}</div>
            <div class="small-muted">Confidence: {confidence_text}</div>
            <div class="small-muted">Engine: {result["engine"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if result.get("rule"):
        st.info(f"Smart correction applied: {result['rule']}")

    if result["engine"] == "Demo mode":
        st.warning("Demo mode is only for UI testing. Train the real model for final prediction.")

    if confidence is not None:
        st.progress(min(max(confidence, 0), 1))

    col1, col2 = st.columns([1.2, 1])

    with col1:
        with st.expander("Cleaned text sent to model", expanded=True):
            st.code(result["cleaned_text"])

    with col2:
        if probabilities is not None:
            with st.expander("Model scores", expanded=True):
                temp = probabilities.copy()
                temp["sentiment"] = temp["sentiment"].apply(sentiment_text)

                st.dataframe(
                    temp,
                    use_container_width=True,
                    hide_index=True
                )

                if HAS_PLOTLY:
                    fig = px.bar(
                        temp,
                        x="sentiment",
                        y="score",
                        title="Sentiment score distribution"
                    )
                    fig.update_layout(height=320)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(temp.set_index("sentiment"))


def get_top_words(series, limit=20):
    stop_words = {
        "the", "and", "for", "you", "that", "this", "with", "from", "are",
        "was", "were", "have", "has", "had", "not", "but", "they", "them",
        "your", "our", "about", "covid", "coronavirus", "https", "http",
        "www", "com", "amp", "will", "all", "can", "just", "people",
        "their", "there", "been", "than", "what", "when", "where", "who"
    }

    text = " ".join(series.dropna().astype(str).str.lower().tolist())
    words = re.findall(r"[a-zA-Z_]{3,}", text)
    words = [word for word in words if word not in stop_words]

    counter = Counter(words)

    return pd.DataFrame(
        counter.most_common(limit),
        columns=["word", "count"]
    )


def render_pipeline():
    steps = [
        ("📥", "Input", "Text comment, audio recording, or CSV file."),
        ("🧹", "Cleaning", "Lowercase, clean text, and handle negation like not happy."),
        ("🔢", "Vectorization", "TF-IDF converts text into numerical features."),
        ("🤖", "Prediction", "SVM predicts positive, negative, or natural sentiment."),
        ("📊", "Dashboard", "Charts, confidence scores, and prediction history."),
    ]

    cols = st.columns(len(steps))

    for col, step in zip(cols, steps):
        icon, title, text = step

        with col:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-icon">{icon}</div>
                    <div class="pipeline-title">{title}</div>
                    <div class="pipeline-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


available_models = get_available_models()
dataset_df, dataset_path = load_dataset()

vectorizer_exists = (MODELS_DIR / "vectorizer.pkl").exists()
models_count = len(available_models)

selected_model = None
model = None
vectorizer = None

if models_count > 0:
    default_index = 0

    if "svm" in available_models:
        default_index = available_models.index("svm")

    selected_model = available_models[default_index]

    try:
        model, vectorizer = load_sentiment_assets(selected_model)
    except Exception:
        model = None
        vectorizer = None


st.sidebar.title("⚙️ Control Panel")

if models_count > 0:
    selected_model = st.sidebar.selectbox(
        "Choose trained model",
        available_models,
        index=available_models.index(selected_model)
    )

    try:
        model, vectorizer = load_sentiment_assets(selected_model)
        st.sidebar.success("Trained model loaded")
    except Exception as e:
        model = None
        vectorizer = None
        st.sidebar.error("Model loading failed")
        st.sidebar.code(str(e))
else:
    st.sidebar.error("No trained model found")
    st.sidebar.caption("The app will run in demo mode.")

whisper_size = st.sidebar.selectbox(
    "Whisper model for audio",
    ["tiny", "base", "small"],
    index=1
)

if st.sidebar.button("Clear cache / reload app"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.subheader("Project Status")

if dataset_df is not None:
    st.sidebar.success("Dataset found")
    st.sidebar.caption(dataset_path)
else:
    st.sidebar.warning("Dataset missing")

if MODELS_DIR.exists():
    st.sidebar.success("models/ folder found")
else:
    st.sidebar.warning("models/ folder missing")

if vectorizer_exists:
    st.sidebar.success("vectorizer.pkl found")
else:
    st.sidebar.warning("vectorizer.pkl missing")

st.sidebar.code("python src/main.py", language="bash")

st.sidebar.divider()

if "history" in st.session_state and len(st.session_state.history) > 0:
    st.sidebar.subheader("Latest Predictions")

    st.sidebar.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Covid Sentiment AI Dashboard</div>
        <div class="hero-subtitle">
            A smart Streamlit dashboard for Covid-related sentiment analysis using text preprocessing,
            negation handling, audio transcription, TF-IDF vectorization, machine learning models,
            and rich analytical visualizations.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


tab_home, tab_predict, tab_audio, tab_dataset, tab_batch, tab_model, tab_about = st.tabs(
    [
        "🏠 Overview",
        "✍️ Text Analysis",
        "🎙️ Audio Analysis",
        "📊 Dataset Dashboard",
        "📁 Batch Analysis",
        "🧪 Model Lab",
        "📘 Project Report"
    ]
)


with tab_home:
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if dataset_df is not None:
            metric_card("Dataset", f"{len(dataset_df):,}", "Rows loaded", "status-ok")
        else:
            metric_card("Dataset", "Missing", "Waiting for CSV", "status-bad")

    with col2:
        if models_count > 0:
            metric_card("Models", models_count, "Trained .pkl files", "status-ok")
        else:
            metric_card("Models", "0", "Demo mode active", "status-warn")

    with col3:
        if vectorizer_exists:
            metric_card("Vectorizer", "Ready", "TF-IDF saved", "status-ok")
        else:
            metric_card("Vectorizer", "Missing", "Train first", "status-warn")

    with col4:
        engine_name = "ML" if model is not None and vectorizer is not None else "Demo"
        metric_card(
            "Prediction Engine",
            engine_name,
            "Current runtime mode",
            "status-ok" if engine_name == "ML" else "status-warn"
        )

    st.markdown("### Pipeline")
    render_pipeline()

    st.markdown("### Current Situation")

    if dataset_df is None and models_count == 0:
        st.error("You need the dataset and trained model files.")
    elif dataset_df is None:
        st.warning("Dataset missing. Prediction can work only if trained models exist.")
    elif models_count == 0:
        st.warning("Dataset exists, but models are not trained yet. Run: python src/main.py")
    else:
        st.success("Everything looks ready.")

    st.markdown("### Dashboard Content")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            """
            <div class="card">
                <h4>📊 Data Analytics</h4>
                <p>Class distribution, tweet lengths, word frequency, samples, and quality checks.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_b:
        st.markdown(
            """
            <div class="card">
                <h4>🎙️ Audio AI</h4>
                <p>Record or upload audio, transcribe it to text, then classify the sentiment.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_c:
        st.markdown(
            """
            <div class="card">
                <h4>📁 Batch Prediction</h4>
                <p>Upload a CSV file, classify many comments at once, visualize results, and export predictions.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


with tab_predict:
    st.subheader("Text Sentiment Analysis")

    examples = [
        "I am not happy with covid restrictions.",
        "I am happy that covid cases are decreasing.",
        "The government announced new covid measures today.",
        "The vaccine helped many people feel safe.",
        "Covid ruined my life and made me anxious."
    ]

    selected_example = st.selectbox("Quick examples", [""] + examples)

    user_text = st.text_area(
        "Write a Covid-related tweet/comment",
        value=selected_example if selected_example else "",
        placeholder="Example: I am not happy with covid restrictions but vaccines are useful.",
        height=170
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        analyze_button = st.button("Analyze Sentiment", use_container_width=True)

    with col2:
        clear_button = st.button("Clear", use_container_width=True)

    if clear_button:
        st.rerun()

    if analyze_button:
        if user_text.strip() == "":
            st.warning("Write a text first.")
        else:
            result = predict_sentiment_ui(user_text, model, vectorizer)
            show_prediction_result(result)
            save_prediction_history("text", user_text, result)


with tab_audio:
    st.subheader("Audio Sentiment Analysis")

    st.info(
        "Record your voice or upload an audio file. The app transcribes it to text using Whisper, "
        "then predicts the sentiment using the same trained model."
    )

    col_record, col_upload = st.columns(2)

    with col_record:
        st.markdown("### Record audio")

        audio_input_func = getattr(st, "audio_input", None)

        if audio_input_func is not None:
            recorded_audio = st.audio_input("Record your Covid opinion")
        else:
            recorded_audio = None
            st.warning("Your Streamlit version does not support audio recording. Use upload instead.")

    with col_upload:
        st.markdown("### Upload audio")
        uploaded_audio = st.file_uploader(
            "Upload WAV, MP3, M4A, or OGG",
            type=["wav", "mp3", "m4a", "ogg"]
        )

    audio_source = recorded_audio if recorded_audio is not None else uploaded_audio

    if audio_source is not None:
        st.audio(audio_source)

    transcribe_button = st.button(
        "Transcribe and Predict Sentiment",
        use_container_width=True
    )

    if transcribe_button:
        if audio_source is None:
            st.warning("Record or upload an audio file first.")
        else:
            try:
                with st.spinner("Transcribing audio with Whisper..."):
                    whisper_model = get_whisper_model(whisper_size)
                    transcribed_text = audio_file_to_text(
                        audio_source,
                        whisper_model,
                        language="en"
                    )

                if transcribed_text is None:
                    st.error("No clear speech detected. Try again.")
                else:
                    st.markdown("### Transcribed text")
                    st.info(transcribed_text)

                    result = predict_sentiment_ui(transcribed_text, model, vectorizer)
                    show_prediction_result(result)
                    save_prediction_history("audio", transcribed_text, result)

            except Exception as e:
                st.error("Audio processing failed.")
                st.exception(e)


with tab_dataset:
    st.subheader("Dataset Dashboard")

    if dataset_df is None:
        st.error("No dataset found.")
        st.code("data/Covid_datasets.csv", language="text")

    else:
        df = dataset_df.copy()

        st.success(f"Dataset loaded from: {dataset_path}")

        detected_text_col, detected_label_col = detect_columns(df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Rows", f"{len(df):,}", "Total records", "status-ok")

        with col2:
            metric_card("Columns", len(df.columns), "Dataset columns", "status-ok")

        with col3:
            missing_count = int(df.isna().sum().sum())
            metric_card(
                "Missing Values",
                missing_count,
                "Total NaN cells",
                "status-warn" if missing_count else "status-ok"
            )

        with col4:
            duplicates = int(df.duplicated().sum())
            metric_card(
                "Duplicates",
                duplicates,
                "Duplicated rows",
                "status-warn" if duplicates else "status-ok"
            )

        st.markdown("### Column Detection")

        c1, c2 = st.columns(2)

        with c1:
            selected_text_col = st.selectbox(
                "Text column",
                df.columns,
                index=list(df.columns).index(detected_text_col)
                if detected_text_col in df.columns
                else 0
            )

        with c2:
            selected_label_col = st.selectbox(
                "Sentiment column",
                ["None"] + list(df.columns),
                index=(["None"] + list(df.columns)).index(detected_label_col)
                if detected_label_col in df.columns
                else 0
            )

        df["_text"] = df[selected_text_col].astype(str)
        df["_text_length"] = df["_text"].str.len()
        df["_word_count"] = df["_text"].str.split().str.len()

        if selected_label_col != "None":
            df["_sentiment"] = df[selected_label_col].apply(normalize_label)

        st.markdown("### Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### Sentiment Distribution")

            if selected_label_col != "None":
                dist = df["_sentiment"].value_counts().reset_index()
                dist.columns = ["sentiment", "count"]

                if HAS_PLOTLY:
                    fig = px.bar(
                        dist,
                        x="sentiment",
                        y="count",
                        title="Number of tweets by sentiment"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(dist.set_index("sentiment"))

                st.dataframe(dist, use_container_width=True, hide_index=True)
            else:
                st.warning("No sentiment column selected.")

        with chart_col2:
            st.markdown("### Text Length Distribution")

            if HAS_PLOTLY:
                fig = px.histogram(
                    df,
                    x="_text_length",
                    nbins=40,
                    title="Tweet/comment length distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df["_text_length"].value_counts().sort_index())

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.markdown("### Word Count Distribution")

            if HAS_PLOTLY:
                fig = px.histogram(
                    df,
                    x="_word_count",
                    nbins=35,
                    title="Number of words per text"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df["_word_count"].value_counts().sort_index())

        with chart_col4:
            st.markdown("### Top Words")

            top_words = get_top_words(df["_text"], limit=20)

            if HAS_PLOTLY:
                fig = px.bar(
                    top_words,
                    x="count",
                    y="word",
                    orientation="h",
                    title="Most frequent words"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(top_words.set_index("word"))


with tab_batch:
    st.subheader("Batch CSV Sentiment Analysis")

    uploaded_csv = st.file_uploader(
        "Upload CSV for batch prediction",
        type=["csv"],
        key="batch_csv"
    )

    if uploaded_csv is not None:
        batch_df = pd.read_csv(uploaded_csv)

        st.markdown("### Uploaded Data")
        st.dataframe(batch_df.head(10), use_container_width=True)

        detected_text_col, _ = detect_columns(batch_df)

        selected_batch_text_col = st.selectbox(
            "Select text column for prediction",
            batch_df.columns,
            index=list(batch_df.columns).index(detected_text_col)
            if detected_text_col in batch_df.columns
            else 0
        )

        run_batch = st.button("Run Batch Prediction", use_container_width=True)

        if run_batch:
            results = []
            progress = st.progress(0)

            texts = batch_df[selected_batch_text_col].astype(str).tolist()

            for i, text in enumerate(texts):
                result = predict_sentiment_ui(text, model, vectorizer)

                results.append({
                    "original_text": text,
                    "cleaned_text": result["cleaned_text"],
                    "predicted_sentiment": sentiment_text(result["label"]),
                    "confidence": None
                    if result["confidence"] is None
                    else round(result["confidence"] * 100, 2),
                    "engine": result["engine"]
                })

                progress.progress((i + 1) / len(texts))

            result_df = pd.DataFrame(results)

            st.markdown("### Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            pred_dist = result_df["predicted_sentiment"].value_counts().reset_index()
            pred_dist.columns = ["sentiment", "count"]

            st.markdown("### Prediction Distribution")

            if HAS_PLOTLY:
                fig = px.pie(
                    pred_dist,
                    names="sentiment",
                    values="count",
                    title="Batch prediction sentiment distribution",
                    hole=0.45
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(pred_dist.set_index("sentiment"))

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results CSV",
                data=csv_bytes,
                file_name="covid_sentiment_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


with tab_model:
    st.subheader("Model Lab")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Selected Model", selected_model if selected_model else "None", "Current model")

    with col2:
        metric_card(
            "Vectorizer",
            "Found" if vectorizer_exists else "Missing",
            "TF-IDF object",
            "status-ok" if vectorizer_exists else "status-warn"
        )

    with col3:
        engine = "Trained model" if model is not None and vectorizer is not None else "Demo mode"
        metric_card(
            "Engine",
            engine,
            "Runtime prediction mode",
            "status-ok" if engine == "Trained model" else "status-warn"
        )

    st.markdown("### Model Files")

    if MODELS_DIR.exists():
        model_files = []

        for file in sorted(MODELS_DIR.glob("*")):
            if file.is_file():
                model_files.append({
                    "file": file.name,
                    "size_kb": round(file.stat().st_size / 1024, 2)
                })

        if model_files:
            st.dataframe(pd.DataFrame(model_files), use_container_width=True, hide_index=True)
        else:
            st.warning("models/ folder exists but contains no files.")
    else:
        st.warning("models/ folder does not exist.")

    st.markdown("### Model Information")

    if model is not None:
        info = {
            "Model type": type(model).__name__,
            "Classes": ", ".join(map(str, getattr(model, "classes_", []))),
            "Has predict_proba": hasattr(model, "predict_proba"),
            "Has decision_function": hasattr(model, "decision_function"),
        }

        st.json(info)
    else:
        st.warning("No trained model loaded.")

    st.markdown("### Dataset Evaluation")

    if model is None or vectorizer is None:
        st.warning("Cannot evaluate because model/vectorizer are missing.")

    elif dataset_df is None:
        st.warning("Cannot evaluate because dataset is missing.")

    else:
        eval_df = dataset_df.copy()
        eval_text_col, eval_label_col = detect_columns(eval_df)

        if eval_text_col is None or eval_label_col is None:
            st.warning("Cannot evaluate because text/sentiment columns were not detected.")

        else:
            try:
                from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

                eval_df = eval_df[[eval_text_col, eval_label_col]].dropna().copy()
                eval_df["_cleaned"] = eval_df[eval_text_col].astype(str).apply(prepare_text_for_model)

                x_eval = vectorizer.transform(eval_df["_cleaned"])
                y_true = eval_df[eval_label_col].apply(normalize_sentiment_value)
                y_pred = model.predict(x_eval)

                acc = accuracy_score(y_true, y_pred)

                metric_card(
                    "Dataset Accuracy Check",
                    f"{acc * 100:.2f}%",
                    "Evaluation on available CSV"
                )

                labels = sorted(list(set(y_true.astype(str)) | set(pd.Series(y_pred).astype(str))))

                cm = confusion_matrix(
                    y_true.astype(str),
                    pd.Series(y_pred).astype(str),
                    labels=labels
                )

                cm_df = pd.DataFrame(cm, index=labels, columns=labels)

                st.markdown("### Confusion Matrix")
                st.dataframe(cm_df, use_container_width=True)

                if HAS_PLOTLY:
                    fig = px.imshow(
                        cm_df,
                        text_auto=True,
                        title="Confusion Matrix"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Classification Report")

                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                st.dataframe(report_df, use_container_width=True)

            except Exception as e:
                st.error("Evaluation failed.")
                st.exception(e)


with tab_about:
    st.subheader("Project Report Content")

    st.markdown(
        """
        ### Project Title

        Covid Sentiment Analysis using Machine Learning

        ### Objective

        The goal of this project is to analyze Covid-related opinions and classify them into three sentiments:

        - Positive
        - Negative
        - Natural

        ### Main Features

        - Text sentiment prediction
        - Audio transcription and prediction
        - Dataset dashboard
        - Sentiment distribution visualization
        - Text length analysis
        - Word frequency analysis
        - Batch CSV prediction
        - Model comparison
        - Confidence score visualization

        ### Machine Learning Pipeline

        1. Load Covid sentiment dataset
        2. Clean text data
        3. Handle negation expressions
        4. Split data into train and test
        5. Convert text to numerical features using TF-IDF
        6. Train machine learning models
        7. Evaluate with accuracy, confusion matrix, and classification report
        8. Save model and vectorizer using Joblib
        9. Deploy interface with Streamlit

        ### Models Used

        - Support Vector Machine
        - Random Forest Classifier

        ### Commands

        ```bash
        python src/main.py
        streamlit run src/app.py
        ```
        """
    )

    st.info(
        "Our application is a complete Covid sentiment analysis dashboard. "
        "It uses preprocessing, negation handling, TF-IDF vectorization, SVM classification, "
        "audio transcription with Whisper, dataset exploration, batch CSV analysis, and model evaluation."
    )