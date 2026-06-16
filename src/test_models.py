from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from data_processing import handle_negation
import joblib


def load_model_vectorizer(model_name):
  model=joblib.load(f"models/{model_name}.pkl")
  vectorizer=joblib.load("models/vectorizer.pkl")
  return model,vectorizer

def predicte_sentiment(text):
  model,vectorizer=load_model_vectorizer('svm')
  text=handle_negation(text)
  text_vectorize=vectorizer.transfrom([text])
  return model.predict(text_vectorize)


def test_model(model,x_test):
  return model.predict(x_test)

def evaluate_model(y_test,ypred):

  print("accuracy:\n",accuracy_score(y_test,ypred))

  print("confusion matrix:\n",confusion_matrix(y_test,ypred))

  print("classification_report:\n",classification_report(y_test,ypred))
