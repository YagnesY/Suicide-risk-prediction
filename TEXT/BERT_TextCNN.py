import sqlite3

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from Config import *
from torch.utils import data
from transformers import BertTokenizer
from transformers import BertModel
import torch

DATABASE = 'weibo_db.db'


def get_db():
    db = sqlite3.connect(DATABASE)
    return db


def close_connection(db):
    db.close()

def read_data(filename, num=None):
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t,l = data.split("\t")
            texts.append(t)
            labels.append(l)
    if num == None:
        return texts,labels
    else:
        return texts[:num],labels[:num]


class Dataset(data.Dataset):
    def __init__(self, type='train'):
        global result1
        super().__init__()
        self.conn = sqlite3.connect('weibo_db.db')  # 连接到你的数据库
        self.cursor = self.conn.cursor()

        if type == 'train':
            result1 =self.cursor.execute("SELECT comment,state FROM train_data")
            # self.cursor.execute("SELECT * FROM train_data_seg")
        elif type == 'test':
            result1=self.cursor.execute("SELECT comment,state FROM test_data")

        self.lines=result1.fetchall()

        # self.data = self.cursor.fetchall()  # 获取所有数据

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self,index):
        text, label =self.lines[index]
        # print(text)
        # print(label)
        if text is None:
            return self.__getitem__((index + 1) % len(self))  # 返回下一个数据
        # if text is None:
        #     return None, None, None
            # 这里可以放置对非 None 文本的处理逻辑
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        if len(input_ids) < MAX_LEN:
            pad_len = (MAX_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len

        return torch.tensor(input_ids[:MAX_LEN]), torch.tensor(mask[:MAX_LEN]), torch.tensor(int(label))

    def close(self):
        self.cursor.close()
        self.conn.close()

# class BERT_TextCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(BERT_MODEL)
#         for name, param in self.bert.named_parameters():
#             param.requires_grad = False
#
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(3, EMBEDDING))
#         self.conv2 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(5, EMBEDDING))
#         self.conv3 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(7, EMBEDDING))
#
#         self.dropout = nn.Dropout(DROP_PROB)
#         self.linear = nn.Linear(HIDDEN_DIM*3, CLASS_NUM)
#
#     def conv_and_pool(self, conv, input):
#         out = conv(input)
#         out = F.relu(out)
#
#         # out.shape[2]: MAX_LEN - kernel_num + 1
#         # out.shape[3]: 1
#         # 池化不会改变形状，最后两维是1，所以降维
#         return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()
#
#     def forward(self, input, mask):
#         # 先输出 (batch, max_len, embedding) = (10, 20, 768), 进行升维，后面做二维卷积
#         out = self.bert(input, mask)[0].unsqueeze(1)   # 得到词向量
#         out1 = self.conv_and_pool(self.conv1, out)
#         out2 = self.conv_and_pool(self.conv2, out)
#         out3 = self.conv_and_pool(self.conv3, out)
#
#         out = torch.cat([out1, out2, out3], dim=1)
#
#         out = self.dropout(out)
#
#         return self.linear(out)

class BERT_BiLSTM(nn.Module):
    def __init__(self):
        super(BERT_BiLSTM, self).__init__()

        # Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        # LSTM layers
        '''
        input_size：输入数据的特征维数，通常就是 embedding_dim (词向量的维度)
        hidden_size：LSTM中隐层的维度
        num_layers：循环神经网络的层数
        bias：用不用偏置，default=True
        batch_first：这个要注意，通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),而batch_first默认是False,所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
        dropout：默认是0，代表不用dropout
        bidirectional：默认是false，代表不用双向LSTM
        '''
        # self.lstm = nn.LSTM(input_size=EMBEDDING, hidden_size=HIDDEN_DIM, num_layers=N_LAYERS, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(EMBEDDING, HIDDEN_DIM, N_LAYERS, batch_first=True, bidirectional=True)

        # dropout layer
        self.dropout = nn.Dropout(DROP_PROB)

        # linear and sigmoid layers
        self.linear = nn.Linear(HIDDEN_DIM * 2, OUTPUT_SIZE)

    '''
    input: shape = [seq_length, batch_size, input_size]的张量
    h_0: shape = [num_layers * num_directions, batch, hidden_size]的张量，它包含了在当前这个batch_size中每个句子的初始隐藏状态，num_layers就是LSTM的层数，如果bidirectional = True,则num_directions = 2,否则就是１，表示只有一个方向
    c_0: 与h_0的形状相同，它包含的是在当前这个batch_size中的每个句子的初始细胞状态。h_0,c_0如果不提供，那么默认是０
    '''
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 2
        # (4, 100, 384)
        hidden = (weight.new(N_LAYERS * number, BATH_SIZE, HIDDEN_DIM).zero_().float(),
                  weight.new(N_LAYERS * number, BATH_SIZE, HIDDEN_DIM).zero_().float()
                  )

        return hidden


    def forward(self, input, mask, hidden):
        # 先输出 (batch, max_len, embedding) = (10, 20, 768), 进行升维，后面做二维卷积
        out = self.bert(input, mask)[0]   # 得到字向量

        lstm_out, (hidden_last, cn_last) = self.lstm(out, hidden)

        # 正向最后一层，最后一个时刻
        hidden_last_L = hidden_last[-2]  # [batch_size, hidden_num]
        # 反向最后一层，最后一个时刻
        hidden_last_R = hidden_last[-1]  # []
        # 进行拼接
        hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)  # [batch_size, hidden_num*2]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        out = self.linear(out)

        return out


if __name__ == "__main__":

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    db = get_db()
    cursor = db.cursor()
    result = cursor.execute("SELECT comment,state FROM train_data").fetchall()
    train_text, train_label = zip(*result)
    test_text, test_label = zip(*cursor.execute("SELECT comment,state FROM test_data").fetchall())

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True)

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=True)

    model = BERT_BiLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    auc_list=[]
    for e in range(1):
        times = 0
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)
            times += 1
            print(f"loss:{loss:.3f}")

            optimizer.zero_grad()  # 梯度初始化为 0
            loss.backward()   # 反向传播求梯度
            optimizer.step()  # 更新所有参数


        # ------------------  Test  ------------------------

        true_lable_list = []
        pred_lable_list = []
        pred_probability_list = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for b, (test_input, test_mask, test_target) in enumerate(test_loader):
            test_input = test_input.to(DEVICE)
            test_mask = test_mask.to(DEVICE)
            test_target = test_target.to(DEVICE)

            test_pred = model(test_input, test_mask)
            test_pred_ = torch.argmax(test_pred, dim=1)
            true_lable_list = test_target.cpu().numpy().tolist()
            pred_lable_list = test_pred_.cpu().numpy().tolist()
            pred_probability_list += torch.softmax(test_pred, dim=1)[:, 1].cpu().numpy().tolist()

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
        f1_score = 2.0 * precision * recall / (precision + recall)
        # Calculate AUC
        auc = roc_auc_score(true_lable_list,pred_probability_list)
        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1_score * 100, '.2f'))
        auc_list.append(format(auc * 100, '.3f'))
        print(accuracy)
        print('---------------------')

    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)
    print(auc_list)





