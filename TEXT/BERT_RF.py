import torch
import csv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer, BertModel
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from Config import *


def read_data(filename, num=None):
    texts = []
    labels = []

    with open(filename, encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            if len(row) >= 8:  # Ensure there are enough columns in the row
                text = row["comment"]  # Assuming the comment is in the last column
                label = int(row["state"])  # Assuming the state is in the 8th column, if the state is 0 or 1

                texts.append(text)
                labels.append(label)

    return texts[:num], labels[:num]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,  # 尝试增加这个值
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        return input_ids, attention_mask, torch.tensor(self.labels[idx], dtype=torch.long)


if __name__ == "__main__":
    TRAIN_PATH = "/kaggle/input/use-data/2_data.csv"
    TEST_PATH = "/kaggle/input/use-data/2_test_data.csv"
    MAX_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    HIDDEN_SIZE = 128
    BERT_MODEL = 'bert-base-uncased'  # Define your BERT model here

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    train_texts, train_labels = read_data(TRAIN_PATH)
    test_texts, test_labels = read_data(TEST_PATH)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    bert_model = BertModel.from_pretrained(BERT_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)

    bert_model.eval()


    def get_bert_features(text):
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.pooler_output
        return features.detach().cpu().numpy()  # 将结果移回CPU


    train_features = [get_bert_features(text) for text in tqdm(train_texts)]
    test_features = [get_bert_features(text) for text in tqdm(test_texts)]

    # 2. 将特征转换为Tensor类型

    train_features_flat = torch.stack([torch.tensor(f.mean(axis=0), dtype=torch.float32) for f in train_features])

    #     test_features_flat = [f.sum(axis=0) for f in test_features]
    test_features_flat = torch.stack([torch.tensor(f.mean(axis=0), dtype=torch.float32) for f in test_features])

    # 3. 使用RF进行训练和预测
    model =DecisionTreeClassifier(random_state=42)
    model.fit(train_features_flat, train_labels)
    predicted_labels = model.predict(test_features_flat)
    #     predicted_labels = model.predict(test_features_flat)
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)

    predicted_probs = model.predict_proba(test_features_flat)[:, 1]  # 获取属于正类的概率


    auc = roc_auc_score(test_labels, predicted_probs)

    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC: {auc:.4f}")