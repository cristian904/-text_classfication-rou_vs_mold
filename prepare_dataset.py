import pandas as pd

def txt_to_xlsx(sample_path, label_path, output_path):
    sample_labels = dict()
    with open(sample_path, encoding="utf-8") as f:
        for line in f:
            sample_labels[line.split("\t")[0].strip()] = [line.split("\t")[1].strip()]

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            sample_labels[line.split("\t")[0].strip()].append(line.split("\t")[1].strip())
    ids = []
    texts = []
    labels = []
    for key, value in sample_labels.items():
        ids.append(int(key))
        texts.append(value[0])
        labels.append(int(value[1]))

    df = pd.DataFrame(data={"ID": ids, "Text": texts, "Label": labels})
    df.to_excel(output_path, index=False)

def analysis(file_path):
    df = pd.read_excel(file_path)
    print("No of samples:", len(df))
    for label in df['Label'].unique():
        print("Label", label, ":", len(df[df['Label'] == label]))

    df['Words'] = df['Text'].apply(lambda x: len(x.split()))
    print("Mean no of words:", df['Words'].mean())


# txt_to_xlsx("./data/train/samples.txt", "./data/train/dialect_labels.txt", "./data/train/dialect_train.xlsx")
# txt_to_xlsx("./data/test/samples.txt", "./data/test/category_labels.txt", "./data/test/category_test.xlsx")
analysis("./data/test/dialect_test.xlsx")