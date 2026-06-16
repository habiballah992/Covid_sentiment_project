import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
import os
import re


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
  vectorizer=TfidfTransformer()
  x_train=vectorizer.fit_transform(x_train)
  x_test=vectorizer.transform(x_test)

  os.makedirs("/models",exist_ok=True)
  joblib.dump(vectorizer,"models/vectorizer.pkl")

  return [x_train,x_test,vectorizer]



def handle_negation(text):
    text = str(text).lower().strip()

    # Convert contractions
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()

    negation_words = ["not", "no", "never", "cannot", "without"]
    modifiers = ["very", "really", "so", "too", "that"]

    new_words = []
    i = 0

    while i < len(words):
        word = words[i]

        if word in negation_words and i + 1 < len(words):

            # case: not very good -> not_very_good
            if words[i + 1] in modifiers and i + 2 < len(words):
                new_words.append(word + "_" + words[i + 1] + "_" + words[i + 2])
                i += 3

            # case: not good -> not_good
            else:
                new_words.append(word + "_" + words[i + 1])
                i += 2

        else:
            new_words.append(word)
            i += 1

    return " ".join(new_words)