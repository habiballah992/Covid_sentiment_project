import sys

from data_processing import load_data, cleaning_data, split_data, vectorize_data
from train import train_model
from test_models import test_model, evaluate_model


df = load_data()

df = cleaning_data(df)

x_train, x_test, y_train, y_test = split_data(df)

x_train_vectorized, x_test_vectorized, vectorizer = vectorize_data(x_train, x_test)

svm = train_model("svm", x_train_vectorized, y_train)

svm_pred = test_model(svm, x_test_vectorized)

print("\n==============================")
print("Model: SVM")
print("==============================")
evaluate_model(y_test, svm_pred)


if "--rf" in sys.argv:
    random_forest = train_model("RandomForest", x_train_vectorized, y_train)
    rf_pred = test_model(random_forest, x_test_vectorized)

    print("\n==============================")
    print("Model: RandomForest")
    print("==============================")
    evaluate_model(y_test, rf_pred)