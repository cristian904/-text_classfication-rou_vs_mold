from string import punctuation

class Preprocessor:

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

    def preprocessing(self,  text, types):
        pipeline = []
        if "stopwords" in types:
            pipeline.append(self.remove_stopword)
        if "punctuation" in types:
            pipeline.append(self.remove_punctuation)
        for f in pipeline:
            text = f(text)
        return text


