# from Config import *
# import torch
# from sklearn import svm
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.feature_extraction.text import CountVectorizer
#
#
# def read_data(filename, num=None):
#     with open(filename, encoding="utf-8") as f:
#         all_data = f.read().split("\n")
#
#     texts = []
#     labels = []
#     for data in all_data:
#         if data:
#             t, l = data.split("\t")
#             texts.append(t)
#             labels.append(int(l))
#     return texts[:num], labels[:num]
#
#
# if __name__ == "__main__":
#     train_text, train_label = read_data(TRAIN_PATH)
#     test_text, test_label = read_data(TEST_PATH)
#
#     vectorizer = CountVectorizer()
#     train_features = vectorizer.fit_transform(train_text)
#     test_features = vectorizer.transform(test_text)
#
#     svm_model = svm.SVC()
#     svm_model.fit(train_features, train_label)
#     predicted_labels = svm_model.predict(test_features)
#
#     accuracy = accuracy_score(test_label, predicted_labels)
#     precision = precision_score(test_label, predicted_labels)
#     recall = recall_score(test_label, predicted_labels)
#     f1 = f1_score(test_label, predicted_labels)
#
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)
from sklearn.feature_extraction.text import CountVectorizer

from Config import *
from sklearn import svm  # 分类器
from sklearn import metrics  # 模型评价， 混淆矩阵


# 导入数据
def read_data(filename, num=None):
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t, l = data.split("\t")
            texts.append(t)
            labels.append(int(l))
    return texts[:num], labels[:num]


if __name__ == "__main__":
    x_train, y_train = read_data(TRAIN_PATH);
    x_test, y_test = read_data(TEST_PATH);
    # 继承类，传入不同的核函数
    # clf_1 = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr')  # 使用rbf径向基函数来讲低维数据转化为高维数据，使其可分
    clf_2 = svm.SVC()
    # clf_3 = svm.SVC(C=1, kernel='poly', gamma=20, decision_function_shape='ovr')
    # clf_4 = svm.SVC(C=1, kernel='sigmoid', gamma=20, decision_function_shape='ovr')
    vectorizer = CountVectorizer();
    train_features = vectorizer.fit_transform(x_train);
    test_features = vectorizer.transform(x_test);
    # 对应不同的核函数， 拟合
    # clf_1.fit(train_features, y_train);
    clf_2.fit(train_features, y_train)
    # clf_3.fit(x_train, y_train)
    # clf_4.fit(x_train, y_train)

    # 预测
    # y_pred_1 = clf_1.predict(test_features);
    y_pred_2 = clf_2.predict(test_features)
    # y_pred_3 = clf_3.predict(x_test)
    # y_pred_4 = clf_4.predict(x_test)

    # 分类评估
    # print(metrics.classification_report(y_test, y_pred_1))
    print(metrics.classification_report(y_test, y_pred_2))
    # print(metrics.classification_report(y_test, y_pred_3))
    # print(metrics.classification_report(y_test, y_pred_4))

    # 混淆矩阵
    # print(metrics.confusion_matrix(y_test, y_pred_1))
    print(metrics.confusion_matrix(y_test, y_pred_2))
    # print(metrics.confusion_matrix(y_test, y_pred_3))
    # print(metrics.confusion_matrix(y_test, y_pred_4))
