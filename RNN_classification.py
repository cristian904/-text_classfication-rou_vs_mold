from keras.models import Sequential
from keras.layers import GRU, Embedding, Dense
from Preprocessor import Preprocessor
DROPOUT = 0.2
MAX_SEQUENCE_LENGTH = 1000
BATCH_SIZE = 100
EPOCHS = 15

word_dict = {}
preprop = Preprocessor()

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

def create_embeddings(X_train):
    X_train = list(map(lambda x: preprop.preprocessing(x), X_train))
    return X_train


# model = create_RNN_model(X_train, y_train, nClasses, DROPOUT, MAX_SEQUENCE_LENGTH, embeddings)
# scores = model.evaluate(X_test, y_test)