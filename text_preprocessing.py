from string import punctuation

class Preprocessing:

    def __init__(self):
        self.stopwords = list(map(lambda x: x.strip(), open("./data/ro_stopwords.txt", "r", encoding="utf-8").readlines()))
        self.punctuation = punctuation.replace("$", "") + "0123456789"

    def remove_stopword(self, text):
        for stopword in self.stopwords:
            text = text.replace(" " + stopword + " ", " ")
            text = text.replace(" " + stopword, " ")
            text = text.replace(stopword + " ", " ")
        return text

    def remove_punctuation(self, text):
        for punct in self.punctuation:
            text = text.replace(punct, "")
        return text

    def preprocessing(self,  text):
        pipeline = []
        for f in pipeline:
            text = f(text)
        return text


