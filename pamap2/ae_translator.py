import read
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

k = 3

results_file = 'ae_translator.csv'


def auto_encoder():
    _input = Input(shape=(180,))
    encoded = Dense(96, activation='sigmoid')(_input)
    decoded = Dense(180)(encoded)
    return Model(inputs=_input, outputs=decoded)


feature_data = read.read()
test_ids = list(feature_data.keys())

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
    hc_train_data = np.concatenate([h_train_data, c_train_data], axis=1)
    print(hc_train_data.shape)
    ca_train_data = np.concatenate([c_train_data, a_train_data], axis=1)
    print(ca_train_data.shape)
    ha_train_data = np.concatenate([h_train_data, a_train_data], axis=1)
    print(ha_train_data.shape)

    h_test_data = np.array(h_test_data)
    c_test_data = np.array(c_test_data)
    a_test_data = np.array(a_test_data)

    ae_model = auto_encoder()
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(h_train_data, a_train_data, verbose=0, epochs=100, shuffle=True)

    a_test_data = ae_model.predict(h_test_data)
    ha_test_data = np.concatenate([h_test_data, a_test_data], axis=1)
    print(ha_test_data.shape)

    ae_model = auto_encoder()
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(c_train_data, a_train_data, verbose=0, epochs=100, shuffle=True)

    a_test_data = ae_model.predict(c_test_data)
    ca_test_data = np.concatenate([c_test_data, a_test_data], axis=1)
    print(ca_test_data.shape)

    ae_model = auto_encoder()
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(h_train_data, c_train_data, verbose=0, epochs=100, shuffle=True)

    c_test_data = ae_model.predict(h_test_data)
    hc_test_data = np.concatenate([h_test_data, c_test_data], axis=1)
    print(ca_test_data.shape)

    cos_acc = read.cos_knn(k, ha_test_data, _test_labels, ha_train_data, _train_labels)
    ha_results.append('ha,'+'a_translator,'+str(k)+',cos_acc,'+str(cos_acc))

    cos_acc = read.cos_knn(k, ca_test_data, _test_labels, ca_train_data, _train_labels)
    ca_results.append('ca,'+'a_translator,'+str(k)+',cos_acc,'+str(cos_acc))

    cos_acc = read.cos_knn(k, hc_test_data, _test_labels, hc_train_data, _train_labels)
    hc_results.append('hc,'+'c_translator,'+str(k)+',cos_acc,'+str(cos_acc))

for item in hc_results:
    read.write_data(results_file, item)
for item in ca_results:
    read.write_data(results_file, item)
for item in ha_results:
    read.write_data(results_file, item)
