from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import fastText as ft
import os
import numpy as np


class FasttextVectorizer(object):

    def __init__(self):
        self.model = None

    def fit(self, texts):
        if os.path.exists("./models/ft_embeddings"):
            self.model = ft.load_model("./models/ft_embeddings")
        else:
            with open("./data/fasttext_input/train.txt", "w", encoding="utf-8") as f:
                [f.write("__label__1" + " " + text + "\n") for text in texts]
            self.model = ft.train_unsupervised("./data/fasttext_input/train.txt", model="skipgram")
            self.model.save_model("./models/ft_embeddings")

    def word_embeddings(self, words):
        embeddings = []
        for word in words:
            embeddings.append(self.model.get_word_vector(word))
        return np.array(embeddings)

    def transform(self, texts):
        new_X_train = []
        for text in texts:
            new_X_train.append(self.model.get_sentence_vector(text))
        return new_X_train

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

class Representation(object):

    @staticmethod
    def get_representation(type):
        if type == "bow":
            return CountVectorizer(ngram_range=(1, 3))
        if type == "tf-idf":
            return TfidfVectorizer()
        if type == "fasttext":
            return FasttextVectorizer()
        return None


class Classifier(object):

    @staticmethod
    def get_classifier(classifier):
        if classifier == "SVM":
            return SVC(kernel='linear')
        if classifier == "LR":
            return LogisticRegression()
        return None


class LoadData(object):

    @staticmethod
    def load_data(type):
        df = pd.read_excel("./data/train/" + type + "_train.xlsx")
        X_train = list(df['Text'])
        y_train = list(df['Label'])
        df = pd.read_excel("./data/test/" + type +  "_test.xlsx")
        X_test = list(df['Text'])
        y_test = list(df['Label'])
        return X_train, list(map(lambda x: x-1, y_train)), X_test, list(map(lambda x: x-1, y_test))