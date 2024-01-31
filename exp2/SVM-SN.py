import sqlite3
from decimal import getcontext, Decimal

import jieba
import pandas as pd
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearnex import patch_sklearn
patch_sklearn()
DATABASE = 'weibo_db.db'


def get_db():
    db = sqlite3.connect(DATABASE)
    return db

def close_connection(db):
    db.close()

def predict_test_data_svm(user):
    db = get_db()
    cursor = db.cursor()

    # Query the training data
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()
    close_connection(db)

    # Convert the result to a DataFrame
    df_train = pd.DataFrame(result, columns=['comment', 'state'])

    # Load Word2Vec model and tokenize the comments
    word2vec_model = Word2Vec.load("word2vec_model.bin")
    df_train['comment'] = df_train['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    # Tokenization and encoding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train['comment'])
    sequences_train = tokenizer.texts_to_sequences(df_train['comment'])
    X_train = pad_sequences(sequences_train, maxlen=180)


    # Build the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(df_train['comment'])

    # Load the labels
    y_train =  np.array(df_train['state'].values.tolist())

    # Train the SVM model

    # svm_model =  SVC(C=0.1,kernel='rbf',probability=True,random_state=42)
    svm_model = BernoulliNB()
    svm_model.fit(X_train, y_train)

    # Query the test data
    db = get_db()
    cursor = db.cursor()
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()
    close_connection(db)

    # Convert the result to a DataFrame
    df_test = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # Tokenize and encode the test comments
    df_test['comment'] = df_test['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    sequences_test = tokenizer.texts_to_sequences(df_test['comment'])
    X_test = pad_sequences(sequences_test, maxlen=180)
    X_test_tfidf = vectorizer.transform(df_test['comment'])

    def softmax(logits):
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Predict using the SVM model
    predictions = svm_model.predict_proba(X_test)
    predictions2 = svm_model.predict(X_test)
    nb_probabilities = softmax(predictions)
    risk = [p[0] for p in nb_probabilities]
    predicted_states = [1 if p[0] >= 0.5 else 0 for p in  nb_probabilities]

    # Save predictions to the database
    db = get_db()
    cursor = db.cursor()
    for i, state in enumerate(predicted_states):
        query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
        parameters = (float(risk[i]), int(df_test['id'][i]))
        cursor.execute(query, parameters)
    db.commit()
    close_connection(db)

    # Calculate metrics
    states = np.array(df_test['state'])
    accuracy = accuracy_score(states,  predictions2)
    precision = precision_score(states,  predictions2 )
    recall = recall_score(states,  predictions2 )
    f1 = f1_score(states,  predictions2 )

    print(accuracy,precision,recall,f1)
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
def build_graph_from_relations(user):
    # 建立数据库连接
    db = get_db()
    cursor = db.cursor()

    # 查询关系数据表
    cursor.execute("SELECT fan_id, follow_id FROM wb_user_relation WHERE user = ?", (user,))
    relation_data = cursor.fetchall()

    # 获取所有用户的total_wb之和
    cursor.execute("SELECT SUM(total_wb) AS total_wb_sum FROM wb_userinfo WHERE user = ?", (user,))
    total_wb_sum = cursor.fetchone()[0]

    # 创建一个空的有向图
    G = nx.DiGraph()

    # 遍历关系数据
    for row in relation_data:
        fan_id = int(row[0])
        follow_id = int(row[1])

        # 查询对应的weibo_id和其他属性值
        cursor.execute(
            "SELECT weibo_id, total_wb, night_wb, followers, following, trisk FROM wb_userinfo WHERE user = ? AND weibo_id = ?",
            (user, fan_id,))
        fan_data = cursor.fetchone()
        fan_weibo_id = int(fan_data[0])
        fan_total_wb = int(fan_data[1])
        fan_night_wb = int(fan_data[2])
        fan_followers = int(fan_data[3])
        fan_following = int(fan_data[4])
        fan_trisk = float(fan_data[5])
        confidence1 = 1/(fan_followers+fan_following)
        night1 = fan_night_wb/50
        # night1 = fan_night_wb / fan_total_wb
        activity1 = fan_total_wb/total_wb_sum

        cursor.execute(
            "SELECT weibo_id, total_wb, night_wb, followers, following, trisk FROM wb_userinfo WHERE user = ? AND weibo_id = ?",
            (user, follow_id,))
        follow_data = cursor.fetchone()
        follow_weibo_id = int(follow_data[0])
        follow_total_wb = int(follow_data[1])
        follow_night_wb = int(follow_data[2])
        follow_followers = int(follow_data[3])
        follow_following = int(follow_data[4])
        follow_trisk = float(follow_data[5])

        confidence = 1 / (follow_followers + follow_following)
        night = follow_night_wb / 50
        # night = follow_night_wb / follow_total_wb
        activity = follow_total_wb / total_wb_sum


        # 添加节点到图中（如果节点不存在）
        if not G.has_node(fan_weibo_id):
            G.add_node(fan_weibo_id, activity=activity1, night=night1, confidence=confidence1, trisk=fan_trisk, prisk=0.0)
        if not G.has_node(follow_weibo_id):
            G.add_node(follow_weibo_id, activity=activity, night=night, confidence=confidence, trisk=follow_trisk, prisk=0.0)

        # 添加边到图中
        G.add_edge(fan_weibo_id, follow_weibo_id)

    # 关闭数据库连接
    close_connection(db)
    nx.write_gexf(G, 'graph1.gexf')
    return G

def pagerank_with_suicide(G, alpha=0.85, personalization=None,
                          max_iter=100, tol=1e-6, nstart=None, weight='weight',
                          dangling=None,a = 0.55):
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = {k: v / s for k, v in nstart.items()}

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise nx.NetworkXError(f"Personalization dictionary "
                                   f"must have a value for every node. "
                                   f"Missing nodes: {missing}")
        s = float(sum(personalization.values()))
        p = {k: v / s for k, v in personalization.items()}

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise nx.NetworkXError(f"Dangling node dictionary "
                                   f"must have a value for every node. "
                                   f"Missing nodes: {missing}")
        s = float(sum(dangling.values()))
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight] * 0.55*G.nodes[nbr]['trisk']
                # x[nbr] += alpha * xlast[n] * W[n][nbr][weight] * ((1 - a) * (G.nodes[nbr]['activity']  + G.nodes[nbr]['night'] + G.nodes[nbr]['confidence'])+ a *
                #                                                   G.nodes[nbr]['trisk'])
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] * 0.55*G.nodes[n]['trisk']
            # x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] * ((1-a) * (G.nodes[n]['activity'] + G.nodes[n]['night'] + G.nodes[n]['confidence']) + a * G.nodes[n]['trisk'])#G.nodes[n]['activity'] + G.nodes[n]['night'] + G.nodes[n]['confidence']) + a * G.nodes[n]['trisk'])

        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
        nx.write_gexf(G, 'g.gexf')
    raise nx.NetworkXError(f"pagerank: power iteration failed to converge "
                           f"in {max_iter} iterations.")


def calculate_node_suicide_risk(G, b, pagerank_dict):
    # 设置Decimal的精度
    getcontext().prec = 200

    for node in G.nodes:
        suicide_risk = Decimal(G.nodes[node]['trisk'])
        pr_sum = Decimal(0)
        pr_sum_neighbors = Decimal(0)

        for neighbor in G.neighbors(node):
            neighbor_suicide = Decimal(G.nodes[neighbor]['trisk'])
            pr_value = Decimal(pagerank_dict[neighbor])
            pr_sum_neighbors += pr_value
            pr_sum += neighbor_suicide * pr_value

        if pr_sum_neighbors != Decimal(0):
            avgrisk = pr_sum / pr_sum_neighbors
        else:
            avgrisk = Decimal(0)
        if avgrisk == 0:
            node_risk = suicide_risk
        else:
            node_risk = Decimal(b) * Decimal(suicide_risk) + Decimal(1 - b) * Decimal(avgrisk)
        G.nodes[node]['prisk'] = node_risk


def save_prisk_to_db(G):
    # 建立数据库连接
    db = get_db()
    cursor = db.cursor()

    # 遍历图G中的所有节点
    for node in G.nodes():
        prisk = G.nodes[node]['prisk']  # 获取prisk值

        # 将prisk保留四位有效数字
        prisk_rounded = round(prisk, 4)
        prisk_str = str(prisk_rounded)

        # 更新节点的prisk值
        cursor.execute("UPDATE wb_userinfo SET prisk = ? WHERE weibo_id = ?", (prisk_str , int(node)))

    # 提交事务并关闭数据库连接
    db.commit()
    close_connection(db)


def calculate_metrics1(user):
    # 建立数据库连接和游标
    db = get_db()
    cursor = db.cursor()

    # 查询特定用户的wb_userinfo表的prisk和state列数据
    cursor.execute("SELECT prisk, state, trisk FROM wb_userinfo WHERE user = ?", (user,))
    data = cursor.fetchall()

    # 将查询结果转换为DataFrame
    df = pd.DataFrame(data, columns=['prisk', 'state', 'trisk'])
    df['prisk'] = df['prisk'].astype(float)
    df['trisk'] = df['trisk'].astype(float)
    df['prisk'] = df.apply(lambda row: min(row['prisk'], row['trisk']), axis=1)
    # df['prisk'] = df.apply(lambda row: (row['prisk'] + row['trisk']) / 2, axis=1)
    # df['prisk_binary'] = df['prisk'].apply(lambda x: 1 if x > 0 else 0)
    predictions = df['prisk'].apply(lambda x: 1 if float(x) >= 0.5 else 0)
    # 计算准确率、精确率、召回率和F1值
    accuracy = accuracy_score(df['state'], predictions)
    precision = precision_score(df['state'], predictions)
    recall = recall_score(df['state'], predictions)
    f1 = f1_score(df['state'], predictions)

    report = classification_report(df['state'], predictions)
    # 打印计算结果
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(report)

    # 关闭数据库连接
    close_connection(db)
    return accuracy, precision, recall, f1
# calculate_metrics('2')


def get_user_data(user):
    # 建立数据库连接
    db = get_db()
    cursor = db.cursor()

    # 查询用户数据
    cursor.execute("SELECT * FROM wb_userinfo WHERE user = ?", (user,))
    data = cursor.fetchall()

    # 关闭数据库连接
    close_connection(db)

    return data


predict_test_data_svm("2")
# print(calculate_metrics("2"))
G=build_graph_from_relations(2)
pagerank_dict = pagerank_with_suicide(G)
calculate_node_suicide_risk(G, 0, pagerank_dict)
save_prisk_to_db(G)
calculate_metrics1('2')
