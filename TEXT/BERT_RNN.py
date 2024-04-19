import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer, BertModel
from Config import *

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
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        return input_ids, attention_mask, torch.tensor(self.labels[idx], dtype=torch.long)


class BertRNNModel(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes, max_len):
        super(BertRNNModel, self).__init__()
        self.bert_model = bert_model
        self.rnn = nn.RNN(bert_model.config.hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        rnn_output, _ = self.rnn(bert_output)
        logits = self.fc(rnn_output[:, -1, :])
        return self.sigmoid(logits)

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

if __name__ == "__main__":
    TRAIN_PATH = "/kaggle/input/use-data/2_data.csv"
    TEST_PATH = "/kaggle/input/use-data/2_test_data.csv"

    train_texts, train_labels = read_data(TRAIN_PATH)
    test_texts, test_labels = read_data(TEST_PATH)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    max_len = 128

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    bert_model = BertModel.from_pretrained(BERT_MODEL)
    model = BertRNNModel(bert_model, hidden_size=128, num_classes=2, max_len=max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        for input_ids, attention_mask, labels in tqdm(train_dataloader, desc="Training"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        true_labels = []
        pred_probs = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(test_dataloader, desc="Testing"):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                true_labels.extend(labels.cpu().numpy())
                pred_probs.extend(outputs.cpu().numpy()[:, 1])

        predicted_labels = [1 if p > 0.5 else 0 for p in pred_probs]
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        auc = roc_auc_score(true_labels, pred_probs)

        print(f"Epoch {epoch + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC: {auc:.4f}")
