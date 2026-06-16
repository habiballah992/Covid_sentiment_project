from data_processing import load_data,cleaning_data,vectorize_data,split_data
from train import train_model
from test_models import test_model,evaluate_model


# 1-loading data
df=load_data()

# 2-cleaning data
df=cleaning_data(df)

# 3-split data
x_train,x_test,y_train,y_test=split_data(df)

# 4- vectorize the data
x_train_vectorized,x_test_vectorized,vectorizer=vectorize_data(x_train,x_test)

# 5- training the models (RandomForest and SVM)
RandomForest=train_model('RandomForest',x_train_vectorized,y_train)
svm=train_model("svm",x_train_vectorized,y_train)

# 6-test the models
R_ypred=test_model(RandomForest,x_test_vectorized)
S_ypred=test_model(svm,x_test_vectorized)

# 7-Evaluation and compare the models
 # RandomForest
print("Model RandomForest:\n")
evaluate_model(y_test,R_ypred)

 #SVM
print("Model SVM:\n")
evaluate_model(y_test,S_ypred)

