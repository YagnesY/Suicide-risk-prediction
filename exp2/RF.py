
import sqlite3
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

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
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())


    # 将文本转换成向量表示

    count_vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    X_train_counts = count_vectorizer.fit_transform(texts)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    y_train = labels

    clf = DecisionTreeClassifier(random_state=1)
    rfc = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='gini',random_state=1)
    clf = clf.fit(X_train_tfidf, y_train)
    rfc = rfc.fit(X_train_tfidf,y_train)



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
    # # 加载训练好的Word2Vec模型
    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    # 对评论进行分词
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())


    # 将文本转换成向量表示
    X_test_counts = count_vectorizer.transform(texts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_test = states
    score_c = clf.score(X_test_tfidf, y_test)
    score_r = rfc.score(X_test_tfidf,y_test)
    print("Single Tree:{}".format(score_c), "Random Forest:{}".format(score_r))
    predictions1 = rfc.predict(X_test_tfidf)
    accuracy = accuracy_score(states, predictions1)
    precision = precision_score(states,predictions1)
    recall = recall_score(states, predictions1)
    f1 = f1_score(states, predictions1)

    print(accuracy, precision, recall, f1)

    predictions2= clf.predict(X_test_tfidf)
    accuracy = accuracy_score(states, predictions2)
    precision = precision_score(states,predictions2)
    recall = recall_score(states, predictions2)
    f1 = f1_score(states, predictions2)
    print(accuracy, precision, recall, f1)




(predict_test_data("2"))



