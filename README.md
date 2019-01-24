
# Results on dialect classification

| Preprocessing  | Text representation   | Classifier    | Accuracy  |
| -------------- | ----------------------|---------------|-----------|
| removed punctuation | Fasttext skipgram | Fasttext          | 93.37% |
| removed punctuation | BoW             | Logistic regression | 92.08% |
| removed punctuation | BoW             | SVM                 | 90.44% |
| removed punctuation | Tf-Idf          | Logistic regression | 90.73% |
| removed punctuation | Tf-Idf          | SVM                 | 92.87% |
| removed punctuation | BoW (1-3 grams) | Logistic regression | 92.99% |