
import sqlite3
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from keras.src.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearnex import patch_sklearn
patch_sklearn()

DATABASE = 'weibo_db.db'


def get_db():
    db = sqlite3.connect(DATABASE)
    return db


def close_connection(db):
    db.close()


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

    # 加载训练好的Word2Vec模型
    word2vec_model = Word2Vec.load("word2vec_model.bin")

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: jieba.lcut(str(x)))  # 分词


    # 将分词结果转换为对应的word2vec向量表示
    vectors = []
    for comment in df['comment']:
        vec = np.zeros(word2vec_model.vector_size)  # 初始化评论向量
        count = 0  # 统计词语数量
        for word in comment:
            if word in word2vec_model.wv:
                vec += word2vec_model.wv[word]
                count += 1
        if count != 0:
            vec /= count  # 取平均值
        vectors.append(vec)

    # 将向量列表转换为NumPy数组
    X_train = np.array(vectors)
    y_train = np.array(df['state'].values.tolist())

    # 定义参数范围
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.1, 1]}

    # # 创建SVM模型
    # svm_model = svm.SVC(random_state=42)
    #
    # # 创建GridSearchCV对象
    # grid_search = GridSearchCV(svm_model, param_grid, cv=3, scoring='accuracy', refit=True)
    # grid_search.fit(X_train, y_train)
    #
    # # 输出最佳参数
    # print("Best Parameters:", grid_search.best_params_)
    #
    # # 输出每个参数组合的详细结果
    # results_df = pd.DataFrame(grid_search.cv_results_)
    # print("Detailed Results:")
    # print(results_df[['params', 'mean_test_score', 'std_test_score']])


    model = svm.SVC(C=1, kernel='rbf',random_state=42)
    # model = svm.SVC(C=20,gamma=10,kernel='rbf',random_state=42)
    # model1= svm.SVC(C=1, kernel='poly', gamma=20, decision_function_shape='ovr',random_state=42)
    # model2 = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr', random_state=42)
    # model3= svm.SVC(C=1, kernel='sigmoid', gamma=20, decision_function_shape='ovr',random_state=42)
    model.fit(X_train, y_train)
    # model1.fit(X_train, y_train)
    # model2.fit(X_train, y_train)
    # model3.fit(X_train, y_train)

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
    df.dropna(inplace=True)

    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: jieba.lcut(str(x)))  # 分词

    # 将分词结果转换为对应的word2vec向量表示
    vectors = []
    for comment in df['comment']:
        vec = np.zeros(word2vec_model.vector_size)  # 初始化评论向量
        count = 0  # 统计词语数量
        for word in comment:
            if word in word2vec_model.wv:
                vec += word2vec_model.wv[word]
                count += 1
        if count != 0:
            vec /= count  # 取平均值
        vectors.append(vec)

    # 将向量列表转换为NumPy数组
    X_test = np.array(vectors)
    y_test = np.array(df['state'].values.tolist())

    # # 输出最佳模型在测试集上的性能
    # best_model = grid_search.best_estimator_
    # predictions = best_model.predict(X_test)
    #
    # # 计算指标
    # accuracy = accuracy_score(y_test, predictions)
    # precision = precision_score(y_test, predictions)
    # recall = recall_score(y_test, predictions)
    # f1 = f1_score(y_test, predictions)
    #
    # print("Best Model Performance on Test Set:")
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)
    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # 设置最大序列长度
    max_sequence_length = 180
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
    risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]
    #
    # # 保存预测结果到数据库
    db = get_db()
    cursor = db.cursor()
    for i, state in enumerate(predicted_states):
        query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
        parameters = (float(risk[i]), int(df['id'][i]))
        print(parameters)
        cursor.execute(query, parameters)
    db.commit()
    close_connection(db)

    # # 计算指标
    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)

    # predictions = model.predict(X_test)
    # # predictions1 = model1.predict(X_test)
    # # predictions2 = model2.predict(X_test)
    # # predictions3 = model3.predict(X_test)
    # # 计算指标
    # accuracy = accuracy_score(y_test, predictions)
    # precision = precision_score(y_test, predictions)
    # recall = recall_score(y_test, predictions)
    # f1 = f1_score(y_test, predictions)
    print(accuracy, precision, recall, f1)

    # 计算指标
    # accuracy = accuracy_score(y_test, predictions1)
    # precision = precision_score(y_test, predictions1)
    # recall = recall_score(y_test, predictions1)
    # f1 = f1_score(y_test, predictions1)
    # print(accuracy, precision, recall, f1)
    #
    # # 计算指标
    # accuracy = accuracy_score(y_test, predictions2)
    # precision = precision_score(y_test, predictions2)
    # recall = recall_score(y_test, predictions2)
    # f1 = f1_score(y_test, predictions2)
    # print(accuracy, precision, recall, f1)
    #
    # # 计算指标
    # accuracy = accuracy_score(y_test, predictions3)
    # precision = precision_score(y_test, predictions3)
    # recall = recall_score(y_test, predictions3)
    # f1 = f1_score(y_test, predictions3)
    # print(accuracy, precision, recall, f1)

    return accuracy, precision, recall, f1

def calculate_metrics(user):
    db = get_db()
    cursor = db.cursor()

    # 查询wb_userinfo表中的weibo_id和state字段
    cursor.execute("SELECT weibo_id, state FROM wb_userinfo WHERE user = ?", (user,))
    userinfo_data = cursor.fetchall()

    # 根据weibo_id查询test_data_seg表中的数据并计算trisk平均值
    metrics_data = []
    for userinfo in userinfo_data:
        weibo_id = userinfo[0]
        state = userinfo[1]

        # 查询对应weibo_id的数据
        cursor.execute("SELECT trisk FROM test_data_seg WHERE user = ? AND weibo_id = ?", (user, weibo_id))
        trisk_data = cursor.fetchall()
        trisk_data = [float(data[0]) for data in trisk_data]

        # 计算trisk平均值
        trisk_avg = np.mean(trisk_data)

        # 更新wb_userinfo表中的trisk字段
        cursor.execute("UPDATE wb_userinfo SET trisk = ? WHERE user = ? AND weibo_id = ?", (trisk_avg, user, weibo_id))
        # 将结果添加到列表中
        metrics_data.append((weibo_id, trisk_avg, state))

    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)

    # 将结果转换为DataFrame
    df = pd.DataFrame(metrics_data, columns=['weibo_id', 'trisk_avg', 'state'])

    # 计算指标
    accuracy = accuracy_score(df['state'], df['trisk_avg'].round())
    precision = precision_score(df['state'], df['trisk_avg'].round())
    recall = recall_score(df['state'], df['trisk_avg'].round())
    f1 = f1_score(df['state'], df['trisk_avg'].round())

    return float(accuracy), float(precision), float(recall), float(f1)

#print(calculate_metrics("2"))

predict_test_data("2")
