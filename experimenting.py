from generate import Representation, Classifier, LoadData
from Preprocessor import Preprocessor
from sklearn.metrics import confusion_matrix
import numpy as np

preprocessor = Preprocessor()
TASK = "dialect"

for REPRESENTATION in ["bow"]:
    for CLASSIFIER in ["LR"]:
        print(REPRESENTATION, CLASSIFIER)

        # load data
        X_train, y_train, X_test, y_test = LoadData.load_data(TASK)

        # preprocess data
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
        accuracy = np.array(list(map(lambda x: 1 if x[0] == x[1] else 0, list(zip(y_test, predictions))))).mean()
        print(accuracy)
        conf_mat = confusion_matrix(y_test, predictions)
        print(conf_mat)