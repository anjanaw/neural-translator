import read_dc_pm as read
import numpy as np
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
    dc_train_data, pm_train_data, _train_labels = read.flatten(_train_data)
    dc_test_data, pm_test_data, _test_labels = read.flatten(_test_data)

    dc_train_data = np.array(dc_train_data)
    dc_train_data = np.reshape(dc_train_data, (dc_train_data.shape[0], dc_train_data.shape[1]*dc_train_data.shape[2]))
    pm_train_data = np.array(pm_train_data)
    pm_train_data = np.reshape(pm_train_data, (pm_train_data.shape[0], pm_train_data.shape[1]*pm_train_data.shape[2]))
    train_data = np.concatenate([dc_train_data, pm_train_data], axis=1)
    print(train_data.shape)

    pm_test_data = np.array(pm_test_data)
    pm_test_data = np.reshape(pm_test_data, (pm_test_data.shape[0], pm_test_data.shape[1]*pm_test_data.shape[2]))

    _dc_test_data = ed_knn(pm_test_data, pm_train_data, dc_train_data)
    test_data = np.concatenate([_dc_test_data, pm_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    ed_results.append('dcpm,'+'ed_translator,'+str(k)+','+str(kk)+',cos_acc,'+str(cos_acc))

    __dc_test_data = cos_knn(pm_test_data, pm_train_data, dc_train_data)
    _test_data = np.concatenate([__dc_test_data, pm_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, _test_data, _test_labels, train_data, _train_labels)
    cos_results.append('dcpm,'+'c_translator,'+str(k)+','+str(kk)+',cos_acc,'+str(cos_acc))


for item in ed_results:
    read.write_data(results_file, item)
for item in cos_results:
    read.write_data(results_file, item)

