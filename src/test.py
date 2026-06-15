from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


def test_model(model,x_test):
  return model.predict(x_test)

def evaluate_model(y_test,ypred):

  print("accuracy:\n",accuracy_score(y_test,ypred))

  print("confusion matrix:\n",confusion_matrix(y_test,ypred))

  print("classification_report:\n",classification_report(y_test,ypred))
