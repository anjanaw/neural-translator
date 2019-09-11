import read
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import heapq

k = 3
kk = 3

results_file = 'knn_translator.csv'


def cos_knn(wrist_test_data, wrist_train_data, thigh_train_data):
    cosine = cosine_similarity(wrist_test_data, wrist_train_data)
    top = [(heapq.nlargest(kk, range(len(i)), i.take)) for i in cosine]
    pred = [[thigh_train_data[j] for j in i[:kk]] for i in top]
    pred = np.array(pred)
    pred = np.average(pred, axis=1)
    return pred


def ed_knn(wrist_test_data, wrist_train_data, thigh_train_data):
    euclid = euclidean_distances(wrist_test_data, wrist_train_data)
    top = [(heapq.nlargest(kk, range(len(i)), i.take)) for i in euclid]
    pred = [[thigh_train_data[j] for j in i[:kk]] for i in top]
    pred = np.array(pred)
    pred = np.average(pred, axis=1)
    return pred


feature_data = read.read()
test_ids = list(feature_data.keys())

for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    w_train_data, t_train_data, _train_labels = read.flatten(_train_data)
    w_test_data, t_test_data, _test_labels = read.flatten(_test_data)

    w_train_data = np.array(w_train_data)
    t_train_data = np.array(t_train_data)
    train_data = np.concatenate([w_train_data, t_train_data], axis=1)
    print(train_data.shape)
    w_test_data = np.array(w_test_data)

    t_test_data = ed_knn(w_test_data, w_train_data, t_train_data)
    test_data = np.concatenate([w_test_data, t_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    results = 'euclid_t_translator,'+str(k)+','+str(kk)+',cos_acc,'+str(cos_acc)
    print(results)
    read.write_data(results_file, results)
