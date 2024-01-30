import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from Config import *

# 1. 读取数据
def read_data(filename, num=None):
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")  # 按行划分

    texts = []
    labels = []
    for data in all_data:
        if data:
            t, l = data.split("\t")
            texts.append(t)
            labels.append(int(l))  # 将标签转换为整数
    return texts[:num], labels[:num]

# 2. 构建词汇表和词嵌入
def build_vocab_and_embedding(train_texts, embedding_dim):
    word_to_index = {"<PAD>": 0, "<UNK>": 1}
    index_to_word = {0: "<PAD>", 1: "<UNK>"}

    # 构建词汇表
    for text in train_texts:
        for word in text.split():  # 这里假设文本是按空格分词的
            if word not in word_to_index:
                index = len(word_to_index)
                word_to_index[word] = index
                index_to_word[index] = word

    # 随机初始化词向量（您可能需要替换为预训练的词向量）
    embedding_matrix = nn.Embedding(len(word_to_index), embedding_dim)

    return word_to_index, index_to_word, embedding_matrix

# 3. 创建数据集类
class TextDatasetLSTM(Dataset):
    def __init__(self, texts, labels, word_to_index, max_sequence_length):
        self.texts = texts
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # 将文本转换为索引序列
        sequence = [self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in text.split()]

        # 填充或截断序列，使其具有相同的长度
        if len(sequence) < self.max_sequence_length:
            sequence += [self.word_to_index["<PAD>"]] * (self.max_sequence_length - len(sequence))
        else:
            sequence = sequence[:self.max_sequence_length]

        return torch.LongTensor(sequence), torch.tensor(label, dtype=torch.long)

# 4. 定义BiLSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, num_classes):
        super(BiLSTMModel, self).__init__()
        self.embedding = embedding_matrix
        self.bilstm = nn.LSTM(embedding_matrix.embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply hidden_size by 2 for bidirectional LSTM

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        output = output[:, -1, :]  # Take the last time step's output
        logits = self.fc(output)
        return logits


# 5. 训练和评估函数
def train_and_evaluate(train_loader, test_loader, model, criterion, optimizer, num_epochs):
    model.train()

    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        lr_scheduler.step()

    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            logits = model(batch_inputs)
            _, predicted = torch.max(logits, 1)
            true_labels.extend(batch_labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    EMBEDDING_DIM = 768
    HIDDEN_SIZE = 128
    NUM_CLASSES = 2
    MAX_SEQUENCE_LENGTH = 100
    BATCH_SIZE = 64
    NUM_EPOCHS = 5

    train_texts, train_labels = read_data(TRAIN_PATH)
    test_texts, test_labels = read_data(TEST_PATH)

    word_to_index, index_to_word, embedding_matrix = build_vocab_and_embedding(train_texts, EMBEDDING_DIM)

    train_dataset = TextDatasetLSTM(train_texts, train_labels, word_to_index, MAX_SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TextDatasetLSTM(test_texts, test_labels, word_to_index, MAX_SEQUENCE_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    lstm_model = BiLSTMModel(embedding_matrix, HIDDEN_SIZE, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    train_and_evaluate(train_loader, test_loader, lstm_model, criterion, optimizer, NUM_EPOCHS)