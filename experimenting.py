from generate import Representation, Classifier, LoadData
from Preprocessor import Preprocessor

preprocessor = Preprocessor()

REPRESENTATION = "bow"
CLASSIFIER = "SVM"
TASK = "dialect"

#load data
X_train, y_train, X_test, y_test = LoadData.load_data(TASK)

#preprocess data
X_train = [preprocessor.preprocessing(text, ["punctuation"]) for text in X_train]
X_test = [preprocessor.preprocessing(text, ["punctuation"]) for text in X_test]

#transform to representation
representation = Representation.get_representation(REPRESENTATION)
X_train = representation.fit_transform(X_train)
X_test = representation.transform(X_test)

#classify
classifier = Classifier.get_classifier(CLASSIFIER)
classifier.fit(X_train, y_train)

#predict
predictions = classifier.predict(X_test)
print(predictions)
