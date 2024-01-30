import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from Config import *
from torch.utils import data
import matplotlib.pyplot as plt


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
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_PATH
        elif type == 'test':
            sample_path = TEST_PATH

        self.lines = open(sample_path, encoding='utf-8').readlines()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        print(len(self.lines))
        return len(self.lines)

    def __getitem__(self, index):
        text, label = self.lines[index].split('\t')
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        if len(input_ids) < MAX_LEN:
            pad_len = (MAX_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len

        return torch.tensor(input_ids[:MAX_LEN]), torch.tensor(mask[:MAX_LEN]), torch.tensor(int(label))


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



# if __name__ == '__main__':
#
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     train_text, train_label = read_data(TRAIN_PATH)
#     test_text, test_label = read_data(TEST_PATH)
#
#     train_dataset = Dataset('train')
#     train_loader = data.DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)
#
#     test_dataset = Dataset('test')
#     test_loader = data.DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)
#
#     model = BERT_BiLSTM().to(DEVICE)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     loss_fn = nn.CrossEntropyLoss()
#
#     loss_list = []
#     times_list = []
#     # model.train()
#     for e in range(EPOCH):
#         times = 0
#         # initialize hidden state：初始化隐藏层大小
#         h = model.init_hidden(BATH_SIZE)
#
#         for b, (input, mask, target) in enumerate(train_loader):
#             input = input.to(DEVICE)
#             mask = mask.to(DEVICE)
#             target = target.to(DEVICE)
#
#             h = tuple([each.data for each in h])
#
#             pred = model(input, mask, h)
#             loss = loss_fn(pred, target)
#             times += 1
#             print(f"loss:{loss:.3f}")
#
#             optimizer.zero_grad()  # 梯度初始化为 0
#             loss.backward()  # 反向传播求梯度
#             optimizer.step()  # 更新所有参数
#
#             times_list.append(times)
#             loss_list.append(loss.detach().numpy())
#
#
#         plt.xlabel('loss')
#         plt.ylabel('batch times')
#         data_dict = {}
#         for i, j in zip(times_list, loss_list):
#             data_dict[i] = j
#         x = [i for i in data_dict.keys()]
#         y = [i for i in data_dict.values()]
#         plt.plot(x, y, label='loss trend')
#         plt.show()
#         # ------------------  Test  ------------------------
#
#         right_num = 0
#         times = 0
#         print('完成')
#         h = model.init_hidden(BATH_SIZE)
#         for b, (test_input, test_mask, test_target) in enumerate(test_loader):
#             test_input = test_input.to(DEVICE)
#             test_mask = test_mask.to(DEVICE)
#             test_target = test_target.to(DEVICE)
#
#             h = tuple([each.data for each in h])
#
#             test_pred = model(test_input, test_mask, h)
#             test_pred_ = torch.argmax(test_pred, dim=1)
#             right_num += int(torch.sum(test_pred_ == test_target))
#             times = times + 1
#             print(times)
#         print(right_num)
#         print(f"acc = {right_num / len(test_text) * 100:.3f}%")
#         print('---------------------')
#         torch.save(model, MODEL_DIR + f'{e}.pth')
#
#             # y_pred = torch.argmax(pred, dim=1)
#             # report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), output_dict=True)
#
#             # with torch.no_grad():    # 不计算梯度，不进行反向传播
#             #     test_input, test_mask, test_target = next(iter(test_loader))
#             #     test_input = test_input.to(DEVICE)
#             #     test_mask = test_mask.to(DEVICE)
#             #     test_target = test_target.to(DEVICE)
#             #     test_pred = model(test_input, test_mask)
#             #     test_pred_ = torch.argmax(test_pred, dim=1)
#             #     rightNum += int(torch.sum(test_pred_==test_target))
#             #     # print('test_pred_:  ', test_pred_)
#             #     # print('test_target:  ', test_target)
#             #     test_report = evaluate(test_pred_.cpu().data.numpy(), test_target.cpu().data.numpy(), output_dict=True)
#
#             # print(rightNum)
#             # print(
#             #     '>> epoch:', e,
#             #     'batch:', b,
#             #     'loss:', round(loss.item(), 5),
#             #     'train_acc:', report['accuracy'],
#             #     'dev_acc:', test_report['accuracy'],
#             # )
#
#             # torch.save(model, MODEL_DIR + f'{e}.pth')

if __name__ == "__main__":

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)

    model = BERT_BiLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for e in range(1):
        times = 0
        h = model.init_hidden(BATH_SIZE)
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            h = tuple([each.data for each in h])

            pred = model(input, mask, h)
            loss = loss_fn(pred, target)
            times += 1
            print(f"loss:{loss:.3f}")

            optimizer.zero_grad()  # 梯度初始化为 0
            loss.backward()   # 反向传播求梯度
            optimizer.step()  # 更新所有参数

        # ------------------  Test  ------------------------

        true_lable_list = []
        pred_lable_list = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        h = model.init_hidden(BATH_SIZE)
        for b, (test_input, test_mask, test_target) in enumerate(test_loader):
            test_input = test_input.to(DEVICE)
            test_mask = test_mask.to(DEVICE)
            test_target = test_target.to(DEVICE)

            h = tuple([each.data for each in h])

            test_pred = model(test_input, test_mask, h)
            test_pred_ = torch.argmax(test_pred, dim=1)
            true_lable_list = test_target.cpu().numpy().tolist()
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
        f1_score = 2.0 * precision * recall / (precision + recall)
        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1_score * 100, '.2f'))
        print(accuracy)
        print('---------------------')

    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)

