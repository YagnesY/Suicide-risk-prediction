from Config import *
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer


def read_data(filename, num=None):
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t, l = data.split("\t")
            texts.append(t)
            labels.append(int(l))
    return texts[:num], labels[:num]


if __name__ == "__main__":
    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)

    rf_model = RandomForestClassifier()
    rf_model.fit(train_features, train_label)
    predicted_labels = rf_model.predict(test_features)

    accuracy = accuracy_score(test_label, predicted_labels)
    precision = precision_score(test_label, predicted_labels)
    recall = recall_score(test_label, predicted_labels)
    f1 = f1_score(test_label, predicted_labels)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
