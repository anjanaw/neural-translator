import numpy as np
import random
import read_dc_pm as read

random.seed(0)
np.random.seed(1)

results_file = 'knn.csv'
k = 5

feature_data = read.read()
test_ids = list(feature_data.keys())
all_results = []
pm_results = []
dc_results = []


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

    dc_test_data = np.array(dc_test_data)
    dc_test_data = np.reshape(dc_test_data, (dc_test_data.shape[0], dc_test_data.shape[1]*dc_test_data.shape[2]))
    pm_test_data = np.array(pm_test_data)
    pm_test_data = np.reshape(pm_test_data, (pm_test_data.shape[0], pm_test_data.shape[1]*pm_test_data.shape[2]))
    test_data = np.concatenate([dc_test_data, pm_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    all_results.append('dcpm,knn,'+str(k)+',cos_acc,'+str(cos_acc))
    dc_cos_acc = read.cos_knn(k, dc_test_data, _test_labels, dc_train_data, _train_labels)
    dc_results.append('dc,knn,'+str(k)+',cos_acc,'+str(dc_cos_acc))
    pm_cos_acc = read.cos_knn(k, pm_test_data, _test_labels, pm_train_data, _train_labels)
    pm_results.append('pm,knn,'+str(k)+',cos_acc,'+str(pm_cos_acc))


for item in all_results:
    read.write_data(results_file, item)
for item in dc_results:
    read.write_data(results_file, item)
for item in pm_results:
    read.write_data(results_file, item)

