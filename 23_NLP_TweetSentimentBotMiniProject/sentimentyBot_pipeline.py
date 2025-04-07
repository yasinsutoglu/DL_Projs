import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


def data_preparation(dataframe, tf_idfVectorizer):
    """
    Data is being prepared for the model,
        "label" variable is passed through Label Encoder
        "tweet" variable is inserted into TF-IDF.

    Parameters
    ----------
        dataframe : dataframe
        tf_idfVectorizer : TF-IDF model

    Returns
    -------
        X : TF-IDF matrice
        y : Dependent variable
    """
    dataframe['tweet'] = dataframe['tweet'].str.lower()


    dataframe["label"].replace(1, value="pozitif", inplace=True)
    dataframe["label"].replace(-1, value="negatif", inplace=True)
    dataframe["label"].replace(0, value="nötr", inplace=True)

    dataframe["label"] = LabelEncoder().fit_transform(dataframe["label"])

    dataframe.dropna(axis=0, inplace=True)
    X = tf_idfVectorizer.fit_transform(dataframe["tweet"])
    y = dataframe["label"]
    # 0 = negatif
    # 1 = nötr
    # 2 = pozitif

    return X, y


def logistic_regression(X, y):
    """
    Logistic Regression model established

    Parameters
    ----------
        X : TF-IDF matrice
        y : Dependent variable

    Returns
    -------
        log_model : model object class

    """
    log_model = LogisticRegression(max_iter=10000).fit(X, y)
    print(cross_val_score(log_model,
                X,
                y,
                scoring="accuracy",
                cv=10).mean())
    
    return log_model


def tweets_21(dataframe_new, tweets):
    """
    Making lowercase of tweets and preparing them for the model

    Parameters
    ----------
        dataframe_new : dataframe containing tweets
        tweets : Variable containing tweets in dataframe_new

    Returns
    -------
        dataframe_new : Dataframe containing edited tweets

    """

    # Converting tweets to lowercase
    dataframe_new[tweets] = dataframe_new[tweets].apply(lambda x: " ".join(x.lower() for x in x.split()))

    return dataframe_new
    

def predict_new_tweet(dataframe_new, log_model, tf_idfVectorizer):
    """
    The emotion of tweets from 2021 was determined using the established Logistic Regression model.
    Predicting (as positive, negative or neutral)

    Parameters
    ----------
        dataframe_new : dataframe containing tweets arranged in the original tweets function to enter the model
        log_model : Logistic Regression model object class
        tf_idfVectorizer

    Returns
    -------
        dataframe_new :Dataframe containing username, tweets and labels predicted by the model  

    """
    tweet_tfidf = tf_idfVectorizer.transform(dataframe_new["tweet"])
    predictions = log_model.predict(tweet_tfidf)
    dataframe_new["label"] = predictions
    return dataframe_new


def main():
    dataframe = pd.read_csv("tweets_labeled.csv")
    tf_idfVectorizer = TfidfVectorizer()
    X, y = data_preparation(dataframe, tf_idfVectorizer)
    log_model = logistic_regression(X, y)
    dataframe_new = pd.read_csv("tweets_21.csv")
    predicted_df = predict_new_tweet(dataframe_new, log_model, tf_idfVectorizer)


if __name__ == "__main__":
    print("The process has started.")
    main()

