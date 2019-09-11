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

for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    w_train_data, t_train_data, _train_labels = read.flatten(_train_data)
    w_test_data, t_test_data, _test_labels = read.flatten(_test_data)

    w_train_data = np.array(w_train_data)
    t_train_data = np.array(t_train_data)
    train_data = np.concatenate([w_train_data, t_train_data], axis=1)
    print(train_data.shape)
    w_test_data = np.array(w_test_data)

    ae_model = auto_encoder()
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(w_train_data, t_train_data, verbose=1, epochs=100, shuffle=True)

    t_test_data = ae_model.predict(w_test_data)
    test_data = np.concatenate([w_test_data, t_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    results = 'ae_t_translator,'+str(k)+',cos_acc,'+str(cos_acc)
    print(results)
    read.write_data(results_file, results)


