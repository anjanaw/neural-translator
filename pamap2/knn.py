import read
import numpy as np

k = 5

results_file = 'knn.csv'

feature_data = read.read()
test_ids = list(feature_data.keys())
all_results = []
h_results = []
c_results = []
a_results = []
hc_results = []
ca_results = []
ha_results = []

for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    h_train_data, c_train_data, a_train_data, _train_labels = read.flatten(_train_data)
    h_test_data, c_test_data, a_test_data, _test_labels = read.flatten(_test_data)

    h_train_data = np.array(h_train_data)
    c_train_data = np.array(c_train_data)
    a_train_data = np.array(a_train_data)
    train_data = np.concatenate([h_train_data, c_train_data, a_train_data], axis=1)
    print(train_data.shape)
    hc_train_data = np.concatenate([h_train_data, c_train_data], axis=1)
    print(hc_train_data.shape)
    ca_train_data = np.concatenate([c_train_data, a_train_data], axis=1)
    print(ca_train_data.shape)
    ha_train_data = np.concatenate([h_train_data, a_train_data], axis=1)
    print(ha_train_data.shape)

    h_test_data = np.array(h_test_data)
    c_test_data = np.array(c_test_data)
    a_test_data = np.array(a_test_data)
    test_data = np.concatenate([h_test_data, c_test_data, a_test_data], axis=1)
    print(test_data.shape)
    hc_test_data = np.concatenate([h_test_data, c_test_data], axis=1)
    print(hc_test_data.shape)
    ca_test_data = np.concatenate([c_test_data, a_test_data], axis=1)
    print(ca_test_data.shape)
    ha_test_data = np.concatenate([h_test_data, a_test_data], axis=1)
    print(ha_test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    all_results.append('knn,'+str(k)+',cos_acc,'+str(cos_acc))
    h_cos_acc = read.cos_knn(k, h_test_data, _test_labels, h_train_data, _train_labels)
    h_results.append('h,knn,'+str(k)+',cos_acc,'+str(h_cos_acc))
    c_cos_acc = read.cos_knn(k, c_test_data, _test_labels, c_train_data, _train_labels)
    c_results.append('c,knn,'+str(k)+',cos_acc,'+str(c_cos_acc))
    a_cos_acc = read.cos_knn(k, a_test_data, _test_labels, a_train_data, _train_labels)
    a_results.append('a,knn,'+str(k)+',cos_acc,'+str(a_cos_acc))
    hc_cos_acc = read.cos_knn(k, hc_test_data, _test_labels, hc_train_data, _train_labels)
    hc_results.append('hc,knn,'+str(k)+',cos_acc,'+str(hc_cos_acc))
    ha_cos_acc = read.cos_knn(k, ha_test_data, _test_labels, ha_train_data, _train_labels)
    ha_results.append('ha,knn,'+str(k)+',cos_acc,'+str(ha_cos_acc))
    ca_cos_acc = read.cos_knn(k, ca_test_data, _test_labels, ca_train_data, _train_labels)
    ca_results.append('ca,knn,'+str(k)+',cos_acc,'+str(ca_cos_acc))


for item in all_results:
    read.write_data(results_file, item)
for item in h_results:
    read.write_data(results_file, item)
for item in c_results:
    read.write_data(results_file, item)
for item in a_results:
    read.write_data(results_file, item)
for item in hc_results:
    read.write_data(results_file, item)
for item in ca_results:
    read.write_data(results_file, item)
for item in ha_results:
    read.write_data(results_file, item)
