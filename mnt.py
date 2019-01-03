import numpy as np
import csv
import os
from scipy import fftpack
import keras
from keras.layers import Dense, BatchNormalization, Input, Lambda, concatenate, Flatten
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers.merge import _Merge
np.random.seed(123)

activityType = ["jogging", "sitting", "standing", "walking", "upstairs", "downstairs"]
idList = range(len(activityType))
activityIdDict = dict(zip(activityType, idList))
data_path = 'C:\\IdeaProjects\\Datasets\\selfback\\merge\\'
test_ids = ['007']

def load_selfback_data(path):
    person_data = {}
    files = os.listdir(path)
    for f in files:
        reader = csv.reader(open(os.path.join(path,f), "r"), delimiter=",")
        _class = f.split('_')[1]
        p = f.split('_')[0]
        temp_data = []
        for row in reader:
            temp_data.append(row)
        activity_data = {}
        if p in person_data:
            activity_data = person_data[p]
            activity_data[_class] = temp_data
        else:
            activity_data[_class] = temp_data
        person_data[p] = activity_data
    return person_data

def holdout_train_test_split(user_data, test_ids):
    train_data = {key:value for key, value in user_data.items() if key not in test_ids}
    test_data = {key:value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data

def extract_features(data, win_len=500):
    for person in data:
        person_data = data[person]
        for activity in person_data:
            df = person_data[activity]
            act = activityIdDict.get(activity)
            ws, ts = split_windows(df, win_len, overlap_ratio=1)
            dct_ws = dct(ws)
            dct_ts = dct(ts)
            labs = [act for i in range(len(ws))]
            if act in classes:
                classes[act][0].extend(dct_ws)
                classes[act][1].extend(dct_ts)
                classes[act][2].extend(labs)
            else:
                classes[act] = [dct_ws, dct_ts, labs]
        people[person] = classes
    return people

def split_windows(data, window_length, overlap_ratio=None):
    w = []
    t = []
    i = 0
    N = len(data)
    increment = int(window_length * overlap_ratio)
    while i + window_length < N:
        start = i
        end = start + window_length
        _w = [a[:3] for a in data[start:end]]
        _t = [a[3:] for a in data[start:end]]
        i = int(i + (increment))
        w.append(_w)
        t.append(_t)
    return w, t

def dct(windows, comps=60):
    dct_window = []
    for tw in windows:
        x = [t[0] for t in tw]
        y = [t[1] for t in tw]
        z = [t[2] for t in tw]

        dct_x = np.abs(fftpack.dct(x, norm='ortho'))
        dct_y = np.abs(fftpack.dct(y, norm='ortho'))
        dct_z = np.abs(fftpack.dct(z, norm='ortho'))

        v = np.array([])
        v = np.concatenate((v, dct_x[:comps]))
        v = np.concatenate((v, dct_y[:comps]))
        v = np.concatenate((v, dct_z[:comps]))

        dct_window.append(v)
    return dct_window

def mlp(x):
    x = Dense(360, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def ae(x):
    x = Dense(96, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(180, name='ae')(x)
    return x

def mlp_hat(x1, x2):
    x1 = Dense(360, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x2 = Dense(360, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x = concatenate([x1, x2])
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(6, activation='softmax', name='mlp')(x)
    return x

###################upper########################
def upper(train_data, test_data):
    upper_input = Input((180,))
    upper_output = ae(upper_input)
    upper_model = Model(inputs=upper_input,outputs=upper_output)
    adam = Adam(lr=0.01)
    upper_model.compile(optimizer=adam,loss='mse')
    upper_model.fit(train_data[0], train_data[1], epochs=10, batch_size=64, verbose=0)
    _test_data = upper_model.predict(test_data[0])

####################lower########################
def lower(train_data, test_data):
    lower_input = Input((180,))
    lower_output = mlp(lower_input)
    lower_model = Model(inputs=lower_input,outputs=lower_output)
    adam = Adam(lr=0.01)
    lower_model.compile(optimizer=adam,loss='categorical_crossentropy')
    lower_model.fit(train_data[0], train_data[2], epochs=10, batch_size=64, verbose=0)
    score = lower_model.evaluate(test_data[0], test_data[2], verbose=0)
    print(score)

##################combine######################
def combine(train_data, test_data):
    upper_input = Input((180,))
    upper_output = ae(upper_input)
    upper_model = Model(inputs=upper_input,outputs=upper_output)

    lower_input = Input((180,))
    lower_output = mlp_hat(lower_input, upper_output)

    adam = Adam(lr=0.01)

    combine_model = Model(inputs=[upper_input, lower_input],outputs=[upper_output, lower_output])
    combine_model.compile(optimizer=adam,loss={'ae':'mse', 'mlp':'categorical_crossentropy'},metrics=['accuracy'])

    train_y = train_data[2]
    test_y = test_data[2]
    _train_y = np_utils.to_categorical(train_y, len(activityType))
    _test_y = np_utils.to_categorical(test_y, len(activityType))

    combine_model.fit([train_data[0], train_data[0]], [train_data[1], _train_y], epochs=10, batch_size=64, verbose=1)
    score = combine_model.evaluate([test_data[0], test_data[0]], [test_data[1], _test_y], verbose=1)
    print(score)

###############################################################
user_data = load_selfback_data(data_path)
user_ids = list(user_data.keys())
for user_id in user_ids:
    print(user_id)
    _train_data, _test_data = holdout_train_test_split(user_data, [user_id])
    _lower_train_data = extract_features(_train_data)
    _lower_test_data = extract_features(_test_data)
    combine(_lower_train_data, _lower_test_data)

