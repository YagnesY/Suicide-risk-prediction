def predict_test_data(user):
  # Load and preprocess your data as before
    db = get_db()
    cursor = db.cursor()
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()
    close_connection(db)
    df = pd.DataFrame(result, columns=['comment', 'state'])
    word2vec_model = Word2Vec.load("word2vec_model.bin")
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    texts = df['comment'].values.tolist()

    # Tokenization and word embedding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = 180
    embedding_dim = word2vec_model.vector_size
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    # Create TextRCNN model
    input_layer = Input(shape=(max_sequence_length,), dtype='int32')
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                input_length=max_sequence_length, trainable=False)(input_layer)
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    conv1d = Conv1D(128, 5, activation='relu')(bi_lstm)
    max_pooling = GlobalMaxPooling1D()(conv1d)
    output_layer = Dense(1, activation='sigmoid')(max_pooling)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Pad sequences and train the model
    sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    states = df['state'].values
    history = model.fit(sequences, states, epochs=5, batch_size=32, validation_split=0.2)

    # Load test data and preprocess it
    db = get_db()
    cursor = db.cursor()
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()
    close_connection(db)
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    db = get_db()
    cursor = db.cursor()
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()
    close_connection(db)
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    sequences = tokenizer.texts_to_sequences(texts)
    states = np.array(df['state'].values.tolist())

    # Padding sequences if needed
    sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Now, you can use this model for prediction on the test data.
    predictions = model.predict(sequences)
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)

    print(accuracy,precision,recall,f1)
  

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

        confidence1 = 1 if fan_followers > 10000 else 0
        night1 = fan_night_wb/50
        activity1 = 1 if fan_total_wb / total_wb_sum * 100 > 1 else fan_total_wb / total_wb_sum * 100


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

        confidence = 1 if follow_followers > 10000 else 0
        night = follow_night_wb / 50
        activity = 1 if follow_total_wb / total_wb_sum * 100 > 1 else follow_total_wb / total_wb_sum * 100


        # 添加节点到图中（如果节点不存在）
        if not G.has_node(fan_weibo_id):
            G.add_node(fan_weibo_id, prisk=0.0)
        if not G.has_node(follow_weibo_id):
            G.add_node(follow_weibo_id, prisk=0.0)

        # 添加边到图中
        G.add_edge(fan_weibo_id, follow_weibo_id)

    # 关闭数据库连接
    close_connection(db)
    nx.write_gexf(G, 'graph1.gexf')
    return G

def pagerank_with_suicide(G, alpha=0.85, personalization=None,
                          max_iter=100, tol=1e-6, nstart=None, weight='weight',
                          dangling=None,a = 0.5):
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

            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] *  G.nodes[n]['trisk']

        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
        nx.write_gexf(G, 'g.gexf')
    raise nx.NetworkXError(f"pagerank: power iteration failed to converge "
                           f"in {max_iter} iterations.")
