from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd


class Representation(object):

    @staticmethod
    def get_representation(type):
        if type == "bow":
            return CountVectorizer(ngram_range=(1, 3))
        if type == "tf-idf":
            return TfidfVectorizer()
        if type == "fasttext":
            return None
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
        return X_train, y_train, X_test, y_test