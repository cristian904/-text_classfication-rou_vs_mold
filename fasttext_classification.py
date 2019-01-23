import fastText as ft
import pandas as pd
from text_preprocessing import Preprocessing

preprocessing = Preprocessing()
df = pd.read_excel("./data/train/dialect_train.xlsx")
with open("./data/fasttext_input/train.txt", "w", encoding="utf-8") as f:
    df.apply(lambda x: f.write("__label__"+str(x['Label']) + " " + preprocessing.preprocessing(x['Text'])+"\n"), axis=1)


model = ft.train_supervised("./data/fasttext_input/train.txt")
df = pd.read_excel("./data/test/dialect_test.xlsx")
df['Predicted label'] = df['Text'].apply(lambda x: int(model.predict(x)[0][0].split("__label__")[1]))
df['Correct'] = df.apply(lambda x: 1 if x['Label'] == x['Predicted label'] else 0, axis=1)
print(df['Correct'].mean())
df.to_excel("./data/test/dialect_test_predict.xlsx", index=False)
