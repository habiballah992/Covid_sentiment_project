import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os


def cleaning_data():
  df=pd.read_csv("data/Covid_datasets.csv")

  #choice the columns wanted
  df=df[['clean_tweet','sentiment']]

  df['clean_tweet']=df['clean_tweet'].astype(str).str.lower().str.strip()

  #drop  duplicates
  df=df.drop_duplicates(subset="clean_tweet")
  #drop null columns
  df.dropna(inplace=True)
  
  #drop empty text
  df=df[df['clean_tweet'].str.strip != ""]


  df['sentiment']=[
    'positive' if s=='pos' 
     else 'negative' if s=='neg' 
     else 'natural' for s in df['sentiment']
     ]
  return df


def split_data(x,y):
  return train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#
def vectorize_data(x_train,x_test):
  vectorizer=CountVectorizer()
  x_train=vectorizer.fit_transform(x_train)
  x_test=vectorizer.transform(x_test)

  os.makedirs("/models",exist_ok=True)
  joblib.dump(vectorizer,"models/vectorizer.pkl")

  return [x_train,x_test,vectorizer]