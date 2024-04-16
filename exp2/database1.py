import random
import sqlite3
import threading
from decimal import getcontext, Decimal

import gensim
import jieba
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, SimpleRNN, Bidirectional, GRU, Flatten, \
    concatenate
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    mean_absolute_error, mean_squared_error, matthews_corrcoef, roc_curve, auc
from keras.models import load_model
from torch.optim import Adam

DATABASE = 'weibo_db.db'

def get_db():
    db = sqlite3.connect(DATABASE)
    return db


def close_connection(db):
    db.close()

def load_glove_embedding(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def train_text_cnn(user):
    # 设置 GPU 设备
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if physical_devices:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #     print("GPU is available")
    # else:
    #     print("GPU is not available")


    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))


    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建Text-CNN模型
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 定义回调函数

    checkpoint = ModelCheckpoint('glove_text_cnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    # 定义回调函数，用于打印损失值
    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

            # # 启动线程执行任务
            # thread = threading.Thread(target=run_tasks_after_epoch)
            # thread.start()
            # # 等待任务完成
            # thread.join()

    # 创建 LossHistory 实例
    loss_history = LossHistory()

    # 训练模型
    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history




# 预测测试集数据并计算指标
def predict_test_data(user):
    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建Text-CNN模型
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_text_cnn_model.h5')

    db = get_db()
    cursor = db.cursor()

    # 查询test_data_seg表中满足条件的数据
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # # 标记化和编码
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #
    # # 设置最大序列长度
    max_sequence_length = 180
    #
    # # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    #
    # # 加载训练好的模型
    # model = load_model('text_cnn_model.h5')
    #
    # # 在测试集上进行预测
    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    # 将预测概率写入数据库
    for i, p in enumerate(predictions):
        # 获取对应记录的ID
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()


    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)

    # risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

# train_text_cnn("2")
# print(predict_test_data("2"))


def train_rnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(SimpleRNN(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_rnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_rnn(user):
    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建Text-CNN模型
    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(SimpleRNN(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_rnn_model.h5')


    db = get_db()
    cursor = db.cursor()

    # 查询test_data_seg表中满足条件的数据
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # # 标记化和编码
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #
    # # 设置最大序列长度
    max_sequence_length = 180
    #
    # # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    #
    # # 加载训练好的模型
    # model = load_model('text_cnn_model.h5')
    #
    # # 在测试集上进行预测
    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    # 将预测概率写入数据库
    for i, p in enumerate(predictions):
        # 获取对应记录的ID
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()


    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)

    # risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc


# train_rnn("2")
# print(predict_test_data_rnn("2"))

from keras.layers import LSTM

def train_lstm(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_lstm_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_lstm(user):
    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建Text-CNN模型
    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_lstm_model.h5')


    db = get_db()
    cursor = db.cursor()

    # 查询test_data_seg表中满足条件的数据
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # # 标记化和编码
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #
    # # 设置最大序列长度
    max_sequence_length = 180
    #
    # # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    #
    # # 加载训练好的模型
    # model = load_model('text_cnn_model.h5')
    #
    # # 在测试集上进行预测
    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    # 将预测概率写入数据库
    for i, p in enumerate(predictions):
        # 获取对应记录的ID
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()


    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)

    # risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

# train_lstm("2")
# print(predict_test_data_lstm("2"))

def train_bilstm(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_bilstm_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_bilstm(user):
    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建Text-CNN模型
    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_bilstm_model.h5')


    db = get_db()
    cursor = db.cursor()

    # 查询test_data_seg表中满足条件的数据
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # # 标记化和编码
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #
    # # 设置最大序列长度
    max_sequence_length = 180
    #
    # # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    #
    # # 加载训练好的模型
    # model = load_model('text_cnn_model.h5')
    #
    # # 在测试集上进行预测
    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    # 将预测概率写入数据库
    for i, p in enumerate(predictions):
        # 获取对应记录的ID
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()


    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)

    # risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc


# train_bilstm("2")
# print(predict_test_data_bilstm("2"))


def train_textrnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))  # 使用GRU替代SimpleRNN，并设置return_sequences=True
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))  # 添加第二个GRU层
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_textrnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history



def predict_test_data_textrnn(user):
    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建Text-CNN模型
    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2,
                  return_sequences=True))  # 使用GRU替代SimpleRNN，并设置return_sequences=True
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))  # 添加第二个GRU层
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_textrnn_model.h5')


    db = get_db()
    cursor = db.cursor()

    # 查询test_data_seg表中满足条件的数据
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # # 标记化和编码
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #
    # # 设置最大序列长度
    max_sequence_length = 180
    #
    # # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    #
    # # 加载训练好的模型
    # model = load_model('text_cnn_model.h5')
    #
    # # 在测试集上进行预测
    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    # 将预测概率写入数据库
    for i, p in enumerate(predictions):
        # 获取对应记录的ID
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()


    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)

    # risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

# train_textrnn("2")
# print(predict_test_data_textrnn("2"))

def train_cnn(user):
    # 设置 GPU 设备
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if physical_devices:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #     print("GPU is available")
    # else:
    #     print("GPU is not available")

    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建纯 CNN 模型
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(Flatten())  # 将多维输入展平为一维
    model.add(Dense(1, activation='sigmoid'))  # 输出层

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 定义回调函数
    checkpoint = ModelCheckpoint('glove_cnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    # 定义回调函数，用于打印损失值
    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    # 创建 LossHistory 实例
    loss_history = LossHistory()

    # 训练模型
    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32,
                        callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_cnn(user):
    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    # glove_model = gensim.models.KeyedVectors.load_word2vec_format("GloVe.bin", binary=True)
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建词嵌入矩阵
    # embedding_dim = glove_model.vector_size
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    # for word, i in word_index.items():
    #     if word in glove_model:
    #         embedding_matrix[i] = glove_model[word]
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])  # 词向量维度
    num_words = min(len(word_index) + 1, len(glove_model))  # 词汇表大小
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 构建Text-CNN模型
    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(Flatten())  # 将多维输入展平为一维
    model.add(Dense(1, activation='sigmoid'))  # 输出层
    model.load_weights('glove_cnn_model.h5')


    db = get_db()
    cursor = db.cursor()

    # 查询test_data_seg表中满足条件的数据
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # # 标记化和编码
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #
    # # 设置最大序列长度
    max_sequence_length = 180
    #
    # # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    #
    # # 加载训练好的模型
    # model = load_model('text_cnn_model.h5')
    #
    # # 在测试集上进行预测
    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    # 将预测概率写入数据库
    for i, p in enumerate(predictions):
        # 获取对应记录的ID
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()


    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)

    # risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

# train_cnn("2")
# print(predict_test_data_cnn("2"))

def train_text_rcnn(user):
    # Load data from database
    db = get_db()
    cursor = db.cursor()
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()
    close_connection(db)

    # Convert result to DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])

    # Tokenize and preprocess the comments
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding sequences
    max_sequence_length = 180
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Load pre-trained word embeddings
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # TextRCNN model
    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                                trainable=False)(input_layer)
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    conv_layer = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(lstm_layer)
    max_pool_layer = GlobalMaxPooling1D()(conv_layer)
    concatenate_layer = concatenate([max_pool_layer, lstm_layer[:, -1, :]])
    output_layer = Dense(64, activation='relu')(concatenate_layer)
    output_layer = Dropout(0.2)(output_layer)
    output_layer = Dense(1, activation='sigmoid')(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint('text_rcnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)
    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    # Train the model
    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32,
                        callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_textrcnn(user):
    db = get_db()
    cursor = db.cursor()

    # 查询train_data_seg表中满足条件的数据
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)

    # 加载训练好的Word2Vec模型
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    # 标记化和编码
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 构建TextRCNN模型
    model = load_model('text_rcnn_model.h5')

    db = get_db()
    cursor = db.cursor()

    # 查询test_data_seg表中满足条件的数据
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # 标记化和编码
    sequences = tokenizer.texts_to_sequences(texts)

    # 进行填充
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # 在测试集上进行预测
    predictions = model.predict(data_sequences)

    # 将预测概率写入数据库
    for i, p in enumerate(predictions):
        # 获取对应记录的ID
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()

    # 关闭数据库连接
    close_connection(db)

    # 计算评价指标
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

train_text_rcnn("2")
# print(predict_test_data_textrcnn("2"))