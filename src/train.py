from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model(model_name, x_train, y_train):
    
    os.makedirs("models", exist_ok=True)
    if model_name=='RandomForest':
      model=RandomForestClassifier()
    elif model_name=='svm':
       model=LinearSVC()

    model.fit(x_train,y_train)
    
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"{model_name} saved successfully.")

    return model