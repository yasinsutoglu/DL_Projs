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
    Veri modele hazırlanmaktadır,
        "label" değişkeni Label Encoderdan geçirilir
        "tweet" değişkeni TF-IDF'e sokulur.

    Parameters
    ----------
        dataframe : dataframe
        tf_idfVectorizer : TF-IDF modeli

    Returns
    -------
        X : TF-IDF matrisi
        y : Bağımlı değişken
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
    Logistic Regression modelinin kurulması

    Parameters
    ----------
        X : TF-IDF Matrisi
        y : bağımlı değişken

    Returns
    -------
        log_model : model nesnesi

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
    tweetlerin lowercase yapılması, modele hazırlanması

    Parameters
    ----------
        dataframe_new : tweetleri içeren dataframe
        tweets : dataframe_new'deki tweetleri içeren değişken

    Returns
    -------
        dataframe_new : düzenlenen tweetleri içeren dataframe

    """

    # tweetlerin küçük harfe çevrilemsi
    dataframe_new[tweets] = dataframe_new[tweets].apply(lambda x: " ".join(x.lower() for x in x.split()))


    return dataframe_new



def predict_new_tweet(dataframe_new, log_model, tf_idfVectorizer):
    """
    2021 yılına ait weetlerin kurulmuş olan Logistic Regression modeli ile duygusunun
    (pozitif, negatif ya da nötr olarak)tahmin edilmesi

    Parameters
    ----------
        dataframe_new : modele girmek üzere original tweets fonksiyonunda düzenlenmiş tweetleri içeren dataframe
        log_model : Logistic Regression model nesnesi
        tf_idfVectorizer

    Returns
    -------
        dataframe_new : username, tweet ve model tarfından tahmin edilen label'ları içeren dataframe

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

