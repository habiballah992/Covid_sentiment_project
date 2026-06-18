from data_processing import load_data, cleaning_data, split_data, vectorize_data
from train import train_model
from test_models import test_model, evaluate_model


df = load_data()
df = cleaning_data(df)

x_train, x_test, y_train, y_test = split_data(df)

x_train_vectorized, x_test_vectorized, vectorizer = vectorize_data(x_train, x_test)


models_to_train = ["svm", "RandomForest"]

for model_name in models_to_train:
    model = train_model(model_name, x_train_vectorized, y_train)
    predictions = test_model(model, x_test_vectorized)

    print("\n==============================")
    print(f"Model: {model_name}")
    print("==============================")
    evaluate_model(y_test, predictions)