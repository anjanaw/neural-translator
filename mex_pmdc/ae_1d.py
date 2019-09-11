import read_dc_pm as read
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

k = 3

results_file = 'ae_1d_translator.csv'


def auto_encoder():
    _input = Input(shape=(1280,))
    encoded = Dense(128, activation='relu')(_input)
    encoded = Dense(48, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(960, activation='sigmoid')(encoded)
    return Model(inputs=_input, outputs=decoded)


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
    train_data = np.concatenate([pm_train_data, dc_train_data], axis=1)
    print(train_data.shape)

    pm_test_data = np.array(pm_test_data)
    pm_test_data = np.reshape(pm_test_data, (pm_test_data.shape[0], pm_test_data.shape[1]*pm_test_data.shape[2]))

    ae_model = auto_encoder()
    ae_model.summary()
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(pm_train_data, dc_train_data, verbose=1, epochs=30, shuffle=True)

    _dc_test_data = ae_model.predict(pm_test_data)
    test_data = np.concatenate([pm_test_data, _dc_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    results = 'dcpm,'+'ae_1d_translator,'+str(k)+',cos_acc,'+str(cos_acc)
    read.write_data(results_file, results)

