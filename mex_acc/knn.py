import numpy as np
import random
import read_acw_act as read

random.seed(0)
np.random.seed(1)

results_file = 'knn.csv'
k = 1

feature_data = read.read()
test_ids = list(feature_data.keys())
all_results = []
w_results = []
t_results = []


for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    t_train_data, w_train_data, _train_labels = read.flatten(_train_data)
    t_test_data, w_test_data, _test_labels = read.flatten(_test_data)

    t_train_data = np.array(t_train_data)
    w_train_data = np.array(w_train_data)
    train_data = np.concatenate([w_train_data, t_train_data], axis=1)
    print(train_data.shape)

    w_test_data = np.array(w_test_data)
    t_test_data = np.array(t_test_data)
    test_data = np.concatenate([w_test_data, t_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    all_results.append('wt,knn,'+str(k)+',cos_acc,'+str(cos_acc))
    w_cos_acc = read.cos_knn(k, w_test_data, _test_labels, w_train_data, _train_labels)
    w_results.append('w,knn,'+str(k)+',cos_acc,'+str(w_cos_acc))
    t_cos_acc = read.cos_knn(k, t_test_data, _test_labels, t_train_data, _train_labels)
    t_results.append('t,knn,'+str(k)+',cos_acc,'+str(t_cos_acc))


for item in all_results:
    read.write_data(results_file, item)
for item in w_results:
    read.write_data(results_file, item)
for item in t_results:
    read.write_data(results_file, item)

