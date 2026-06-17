from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"


def train_model(model_name, x_train, y_train):
    MODELS_DIR.mkdir(exist_ok=True)

    model_key = model_name.lower()

    if model_key == "svm":
        model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        )
        save_name = "svm"

    elif model_key == "randomforest":
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=80,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        save_name = "RandomForest"

    else:
        raise ValueError("model_name must be 'svm' or 'RandomForest'")

    print(f"\nTraining {save_name}...")
    model.fit(x_train, y_train)

    model_path = MODELS_DIR / f"{save_name}.pkl"
    joblib.dump(model, model_path)

    print(f"{save_name} saved successfully: {model_path}")

    return model