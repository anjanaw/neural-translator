import read
import numpy as np

k = 3

results_file = 'knn.csv'

feature_data = read.read()
test_ids = list(feature_data.keys())

for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    w_train_data, t_train_data, _train_labels = read.flatten(_train_data)
    w_test_data, t_test_data, _test_labels = read.flatten(_test_data)
    _train_labels
    w_train_data = np.array(w_train_data)
    t_train_data = np.array(t_train_data)
    train_data = np.concatenate([w_train_data, t_train_data], axis=1)
    print(train_data.shape)
    w_test_data = np.array(w_test_data)
    t_test_data = np.array(t_test_data)
    test_data = np.concatenate([w_test_data, t_test_data], axis=1)
    print(test_data.shape)

    # cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    # results = 'knn,'+str(k)+',cos_acc,'+str(cos_acc)
    # w_cos_acc = read.cos_knn(k, w_test_data, _test_labels, w_train_data, _train_labels)
    # results = 'w,knn,'+str(k)+',cos_acc,'+str(w_cos_acc)
    t_cos_acc = read.cos_knn(k, t_test_data, _test_labels, t_train_data, _train_labels)
    results = 't,knn,'+str(k)+',cos_acc,'+str(t_cos_acc)
    print(results)
    read.write_data(results_file, results)
