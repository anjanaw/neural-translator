import read_acw_act as read
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import heapq

k = 3
kk = 1

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

ed_results = []
cos_results = []

for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    t_train_data, w_train_data, _train_labels = read.flatten(_train_data)
    t_test_data, w_test_data, _test_labels = read.flatten(_test_data)

    t_train_data = np.array(t_train_data)
    w_train_data = np.array(w_train_data)
    tw_train_data = np.concatenate([w_train_data, t_train_data], axis=1)
    print(tw_train_data.shape)

    t_test_data = np.array(t_test_data)
    w_test_data = np.array(w_test_data)

    t_test_data = ed_knn(w_test_data, w_train_data, t_train_data)
    tw_test_data = np.concatenate([w_test_data, t_test_data], axis=1)
    print(tw_test_data.shape)

    cos_acc = read.cos_knn(k, tw_test_data, _test_labels, tw_train_data, _train_labels)
    ed_results.append('tw,'+'ed_translator,'+str(k)+','+str(kk)+',cos_acc,'+str(cos_acc))

    t_test_data = cos_knn(w_test_data, w_train_data, t_train_data)
    tw_test_data = np.concatenate([w_test_data, t_test_data], axis=1)
    print(tw_test_data.shape)

    cos_acc = read.cos_knn(k, tw_test_data, _test_labels, tw_train_data, _train_labels)
    cos_results.append('tw,'+'c_translator,'+str(k)+','+str(kk)+',cos_acc,'+str(cos_acc))


for item in ed_results:
    read.write_data(results_file, item)
for item in cos_results:
    read.write_data(results_file, item)

