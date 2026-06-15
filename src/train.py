from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def Training_models(x_train,y_train):
  
  svm=LinearSVC()
  svm.fit(x_train,y_train)

  random_froest=RandomForestClassifier()
  random_froest.fit(x_train,y_train)