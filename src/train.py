from src.data_processing import cleaning_data, split_data, vectorize_data

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
import os


def train_and_evaluate(model, model_name, x_train, x_test, y_train, y_test):
    print(f"\nTraining {model_name}...")

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")

    print(f"{model_name} saved successfully.")


def main():
    df = cleaning_data()

    x = df["clean_tweet"]
    y = df["sentiment"]

    x_train, x_test, y_train, y_test = split_data(x, y)

    x_train, x_test, vectorizer = vectorize_data(x_train, x_test)

    svm_model = LinearSVC()

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    train_and_evaluate(
        svm_model,
        "svm_model",
        x_train,
        x_test,
        y_train,
        y_test
    )

    train_and_evaluate(
        random_forest_model,
        "random_forest_model",
        x_train,
        x_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()