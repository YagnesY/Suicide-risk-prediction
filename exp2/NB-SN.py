

import sqlite3
import jieba
import numpy as np
import networkx as nx
import pandas as pd
from decimal import getcontext, Decimal
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

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

    model =MultinomialNB()
    model.fit(X_train_tfidf, y_train)

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
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    # sequences_test = tokenizer.texts_to_sequences(df['comment'])
    # X_test = pad_sequences(sequences_test, maxlen=180)
    # X_test_tfidf = vectorizer.transform(df['comment'])
    #
    # # Predict using the SVM model
    # predictions = svm_model.predict_proba(X_test_tfidf)[:, 1]
    # predicted_states = [1 if p >= 0.5 else 0 for p in predictions]
    #
    # # Save predictions to the database
    # db = get_db()
    # cursor = db.cursor()
    # for i, state in enumerate(predicted_states):
    #     query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
    #     parameters = (float(predictions[i]), int(df['id'][i]))
    #     cursor.execute(query, parameters)
    # db.commit()
    # close_connection(db)
    #
    # # Calculate metrics
    # states = np.array(df['state'])
    # accuracy = accuracy_score(states, predicted_states)
    # precision = precision_score(states, predicted_states)
    # recall = recall_score(states, predicted_states)
    # f1 = f1_score(states, predicted_states)
    #
    # print(accuracy, precision, recall, f1)
    # 加载数据集
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # 将文本转换成向量表示
    X_test_counts = count_vectorizer.transform(texts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_test = states



    predictions = model.predict(X_test_tfidf)
    # risk = [p[0] for p in predictions]
    # predicted_states = [1 if p[0] >= 0.5 else 0 for p in predictions]
    #
    # # Save predictions to the database
    # db = get_db()
    # cursor = db.cursor()
    # for i, state in enumerate(predicted_states):
    #     query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
    #     parameters = (float(risk[i]), int(df['id'][i]))
    #     cursor.execute(query, parameters)
    # db.commit()
    # close_connection(db)

    accuracy = accuracy_score(states, predictions)
    precision = precision_score(states, predictions)
    recall = recall_score(states, predictions)
    f1 = f1_score(states, predictions)
    print(accuracy, precision, recall, f1)
    return accuracy, precision, recall, f1

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
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight] * G.nodes[nbr]['trisk']
                # x[nbr] += alpha * xlast[n] * W[n][nbr][weight] * ((1 - a) * (G.nodes[nbr]['activity']  + G.nodes[nbr]['night'] + G.nodes[nbr]['confidence'])+ a *
                #                                                   G.nodes[nbr]['trisk'])
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] * G.nodes[n]['trisk']
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

    # 打印计算结果
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # 关闭数据库连接
    close_connection(db)
    return accuracy, precision, recall, f1
# predict_test_data("2")
G=build_graph_from_relations(2)
pagerank_dict = pagerank_with_suicide(G)
calculate_node_suicide_risk(G, 0.7, pagerank_dict)
save_prisk_to_db(G)
calculate_metrics1('2')