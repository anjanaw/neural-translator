import read_dc_pm as read
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, Reshape
from keras.models import Model

k = 3
results_file = 'ae_2D_translator.csv'


def ae_model():
    _input = Input(shape=(16, 16, 1))
    # encoder
    x = Conv2D(32, (3,3), activation='relu', padding='same')(_input)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(4*4*128, activation='sigmoid')(x)
    x = Reshape((4, 4, 128))(x)
    # decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (3, 1), activation='relu', padding='valid')(x)

    x = Conv2D(1, (3, 1), activation='sigmoid', padding='valid')(x)

    model = Model(inputs=_input, outputs=x)
    model.summary()
    return model


feature_data = read.read()
test_ids = list(feature_data.keys())


for test_id in test_ids:
    _train_data, _test_data = read.split(feature_data, [test_id])
    dc_train_data, pm_train_data, _train_labels = read.flatten(_train_data)
    dc_test_data, pm_test_data, _test_labels = read.flatten(_test_data)

    dc_train_data = np.array(dc_train_data)
    print(dc_train_data.shape)
    dc_train_data = np.reshape(dc_train_data, (dc_train_data.shape[0]*dc_train_data.shape[1], 12, 16, 1))
    pm_train_data = np.array(pm_train_data)
    print(pm_train_data.shape)
    pm_train_data = np.reshape(pm_train_data, (pm_train_data.shape[0]*pm_train_data.shape[1], 16, 16, 1))

    pm_test_data = np.array(pm_test_data)
    pm_test_data = np.reshape(pm_test_data, (pm_test_data.shape[0]*pm_test_data.shape[1], 16, 16, 1))

    ae_model = ae_model()
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(pm_train_data, dc_train_data, batch_size=128, verbose=1, epochs=30, shuffle=True)

    dc_test_data = ae_model.predict(pm_test_data)

    dc_train_data = np.reshape(dc_train_data, (dc_train_data.shape[0]/5, 5*12*16))
    pm_train_data = np.reshape(pm_train_data, (pm_train_data.shape[0]/5, 5*16*16))

    train_data = np.concatenate([dc_train_data, pm_train_data], axis=1)
    print(train_data.shape)

    dc_test_data = np.reshape(dc_test_data, (dc_test_data.shape[0]/5, 5*12*16))
    pm_test_data = np.reshape(pm_test_data, (pm_test_data.shape[0]/5, 5*16*16))

    test_data = np.concatenate([dc_test_data, pm_test_data], axis=1)
    print(test_data.shape)

    cos_acc = read.cos_knn(k, test_data, _test_labels, train_data, _train_labels)
    results = 'dcpm,'+'ae_2d_translator,'+str(k)+',cos_acc,'+str(cos_acc)
    read.write_data(results_file, results)

