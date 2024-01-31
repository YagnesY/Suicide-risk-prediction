
import torch
import gensim
import csv
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def read_data(filename, num=None):
    texts = []
    labels = []

    with open(filename, encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        next(csv_reader)  # 跳过标题行

        for row in csv_reader:
            if len(row) >= 8:  # 确保行中有足够的列
                text = row["comment"]  # 假设评论在最后一列
                label = int(row["state"])  # 假设状态在第8列，如果状态是0或1

                texts.append(text)
                labels.append(label)

    return texts[:num], labels[:num]

# 加载预训练的Word2Vec 模型
word2vec_model = gensim.models.Word2Vec.load("your_word2vec_model.model")

# 构建词汇表和词嵌入矩阵
def built_curpus(train_texts):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    embedding_matrix = [np.zeros(word2vec_model.vector_size), np.random.rand(word2vec_model.vector_size)]  # PAD 和 UNK 的向量
    for text in train_texts:
        for word in text.split():
            if word in word2vec_model.wv:
                if word not in word_2_index:
                    word_2_index[word] = len(word_2_index)
                    embedding_matrix.append(word2vec_model.wv.get_vector(word))
    embedding_matrix = np.array(embedding_matrix)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    embedding_matrix = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=False)

    return word_2_index, embedding_matrix

# 修改 TextDataset 类
class TextDataset(Dataset):
    def __init__(self, all_text, all_label, word_2_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_label[index])

        text_idx = [self.word_2_index.get(word, 1) for word in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))
        text_idx = text_idx[:self.max_len]
        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)

        return text_idx, label

    # 返回数据集的样本数量，该方法将在数据加载器中被调用。
    def __len__(self):
        return len(self.all_text)

class CNNModel(nn.Module):
    def __init__(self, max_len, class_num, hidden_num):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(1, 32)).to(DEVICE, dtype=torch.float32)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(1, 64)).to(DEVICE, dtype=torch.float32)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(1, 128)).to(DEVICE, dtype=torch.float32)

        self.dropout = nn.Dropout(DROP_PROB)
        self.linear = nn.Linear(hidden_num * 3, class_num)

    def forward(self, batch_idx, batch_label=None):
        # Convolution and pooling
        batch_idx = batch_idx.unsqueeze(1)  # Add a channel dimension
        out1 = F.relu(self.conv1(batch_idx))
        out2 = F.relu(self.conv2(batch_idx))
        out3 = F.relu(self.conv3(batch_idx))

        out1 = F.max_pool2d(out1, (out1.shape[2], out1.shape[3])).squeeze()
        out2 = F.max_pool2d(out2, (out2.shape[2], out2.shape[3])).squeeze()
        out3 = F.max_pool2d(out3, (out3.shape[2], out3.shape[3])).squeeze()

        feature = torch.cat([out1, out2, out3], dim=1)  # Concatenate feature maps

        feature = self.dropout(feature)
        pred = self.linear(feature)

        return pred


EPOCH = 5
TRAIN_PATH = "1_trian_data.csv"
TEST_PATH = "1_test_data.csv"
BERT_PAD_ID = 0
MAX_LEN = 200       # 最大长度，超过截取，缺少填充
BATH_SIZE = 128
LR = 0.0001         # 学习率
CLASS_NUM = 2      # 分类数，二分类
EMBEDDING = 768    # 子向量维度，BERT 中默认


OUTPUT_SIZE = 2    # BiLSTM 参数
N_LAYERS = 2       # BiLSTM 的层数
HIDDEN_DIM = 128   # LSTM 中隐层的维度，768/2
DROP_PROB = 0.2    # dropout 参数


if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 判断 gpu 是否可用，不可用就是使用 cpu




    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    # 训练自己的Word2Vec 模型
    # sentences = [text.split() for text in train_text]  # train_text 是您的训练文本数据
    # word2vec_model = gensim.models.Word2Vec(sentences, vector_size=EMBEDDING, window=5, min_count=1, workers=4)
    # word2vec_model.save("your_word2vec_model.model")

    #words_embedding：词嵌入模型
    word_2_index, words_embedding = built_curpus(train_text)

    train_dataset = TextDataset(train_text, train_label, word_2_index, MAX_LEN)
    train_loader = DataLoader(train_dataset, BATH_SIZE, shuffle=True)  # 不随机抓取 200 个样本数据

    test_dataset = TextDataset(test_text, test_label, word_2_index, MAX_LEN)
    test_loader = DataLoader(test_dataset, BATH_SIZE, shuffle=True)

    model = CNNModel(MAX_LEN, CLASS_NUM, HIDDEN_DIM).to(DEVICE, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for e in range(EPOCH):
        for batch_idx, batch_label in train_loader:
            batch_idx = batch_idx.to(DEVICE, dtype=torch.float32)
            batch_label = batch_label.to(DEVICE)
            train_pred = model.forward(batch_idx)  # 得到全部特征向量
            loss = loss_fn(train_pred, batch_label)  # 计算损失值

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

        true_lable_list = []
        pred_lable_list = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for batch_idx, batch_label in test_loader:
            batch_idx = batch_idx.to(DEVICE, dtype=torch.float32)
            batch_label = batch_label.to(DEVICE)
            test_pred = model.forward(batch_idx)
            test_pred_ = torch.argmax(test_pred, dim=1)  # 寻找参数最大值的 idx

            true_lable_list = batch_label.cpu().numpy().tolist()
            pred_lable_list = test_pred_.cpu().numpy().tolist()
            for i in range(len(true_lable_list)):
                if pred_lable_list[i] == 0 and true_lable_list[i] == 0:
                    TP += 1
                if pred_lable_list[i] == 1 and true_lable_list[i] == 1:
                    TN += 1
                if pred_lable_list[i] == 0 and true_lable_list[i] == 1:
                    FP += 1
                if pred_lable_list[i] == 1 and true_lable_list[i] == 0:
                    FN += 1
        accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        precision = TP * 1.0 / (TP + FP) * 1.0
        recall = TP * 1.0 / (TP + FN)
        f1score = 2.0 * precision * recall / (precision + recall)
        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1score * 100, '.2f'))
        print(accuracy)
        print('---------------------')

    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)


