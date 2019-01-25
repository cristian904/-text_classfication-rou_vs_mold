from keras.models import Sequential
from keras.layers import GRU, Embedding, Dense
from keras.preprocessing.text import Tokenizer
from generate import LoadData, Representation
import numpy as np
from Preprocessor import Preprocessor
DROPOUT = 0.2
MAX_SEQUENCE_LENGTH = 1000
BATCH_SIZE = 100
EPOCHS = 15
NCLASSES = 2


def create_RNN_model(X_train, y_train, nClasses, DROPOUT, MAX_SEQUENCE_LENGTH, embeddings):
    model = Sequential()
    embedding_layer = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], input_length=MAX_SEQUENCE_LENGTH, trainable=True, name='embedding')
    model.add(embedding_layer)
    model.add(GRU(200, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model

def create_embeddings(X_train, word_dict):
    representation = Representation.get_representation("fasttext")
    representation.fit(X_train)
    embeddings = np.array(np.array(representation.word_embeddings(word_dict.keys())))
    return embeddings


X_train, y_train, X_test, y_test = LoadData.load_data("dialect")
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)
X_train_tokenized = tokenizer.texts_to_matrix(X_train)
word_dict = tokenizer.word_index
embeddings = create_embeddings(X_train, word_dict)
X_test_tokenized = tokenizer.texts_to_matrix(X_train)

model = create_RNN_model(X_train_tokenized, y_train, NCLASSES, DROPOUT, MAX_SEQUENCE_LENGTH, embeddings)
scores = model.evaluate(X_test_tokenized, y_test)
print(scores)