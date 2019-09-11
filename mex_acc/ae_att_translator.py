import read_acw_act as read
import numpy as np
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model

k = 3

results_file = 'ae_att_translator.csv'


def lstm_seq_seq():
    _input = Input(shape=(5, 100))
    x = LSTM(150)(_input)
    x = RepeatVector(5)(x)
    x = LSTM(150, return_sequences=True)(x)
    x = TimeDistributed(Dense(100, activation='softmax'))(x)
    model = Model(inputs=_input, outputs=x)
    model.summary()
    return model


feature_data = read.read()
test_ids = list(feature_data.keys())

for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    t_train_data, w_train_data, _train_labels = read.flatten(_train_data)
    t_test_data, w_test_data, _test_labels = read.flatten(_test_data)

    w_train_data = np.array(w_train_data)
    w_train_data = np.reshape(w_train_data, (w_train_data.shape[0], read.window, read.frames_per_second,
                                             w_train_data.shape[2]))

    wx_train_data = w_train_data[:, :, :, 0]
    wy_train_data = w_train_data[:, :, :, 1]
    wz_train_data = w_train_data[:, :, :, 2]
    print(wz_train_data.shape)

    t_train_data = np.array(t_train_data)
    t_train_data = np.reshape(t_train_data, (t_train_data.shape[0], read.window, read.frames_per_second,
                                             t_train_data.shape[2]))
    tx_train_data = t_train_data[:, :, :, 0]
    ty_train_data = t_train_data[:, :, :, 1]
    tz_train_data = t_train_data[:, :, :, 2]
    print(t_train_data.shape)

    w_test_data = np.array(w_test_data)
    w_test_data = np.reshape(w_test_data, (w_test_data.shape[0], read.window, read.frames_per_second,
                                             w_test_data.shape[2]))

    wx_test_data = w_test_data[:, :, :, 0]
    wy_test_data = w_test_data[:, :, :, 1]
    wz_test_data = w_test_data[:, :, :, 2]
    print(wz_test_data.shape)

    x_model = lstm_seq_seq()
    x_model.compile(optimizer='adam', loss='categorical_crossentropy')
    x_model.fit(wx_train_data, tx_train_data, verbose=1, epochs=10, shuffle=True)

    y_model = lstm_seq_seq()
    y_model.compile(optimizer='adam', loss='categorical_crossentropy')
    y_model.fit(wy_train_data, ty_train_data, verbose=1, epochs=10, shuffle=True)

    z_model = lstm_seq_seq()
    z_model.compile(optimizer='adam', loss='categorical_crossentropy')
    z_model.fit(wz_train_data, tz_train_data, verbose=1, epochs=10, shuffle=True)

    tx_test_data = x_model.predict(wx_test_data)
    tx_test_data = np.expand_dims(tx_test_data, 3)
    ty_test_data = x_model.predict(wy_test_data)
    ty_test_data = np.expand_dims(ty_test_data, 3)
    tz_test_data = x_model.predict(wz_test_data)
    tz_test_data = np.expand_dims(tz_test_data, 3)
    print(tz_test_data.shape)

    t_test_data = np.concatenate([tx_test_data, ty_test_data, tz_test_data], axis=3)
    print(t_test_data.shape)
    print(t_test_data[0])

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    results = 'loss:bce,ae_t_translator,'+str(k)+',cos_acc,'+str(cos_acc)
    print(results)
    read.write_data(results_file, results)


