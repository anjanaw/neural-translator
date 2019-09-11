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
np.random.seed(1)

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
idList = range(len(classes))
activityIdDict = dict(zip(classes, idList))

data_path = 'C:\\IdeaProjects\\Datasets\\selfback\\activity_data_34_9_\\'

def read_data(path):
    person_data = {}
    files = os.listdir(path)
    for f in files:
        temp = f.split("_")
        user = temp[0]
        activity = temp[1]
        data = []
        reader = csv.reader(open(os.path.join(path, f), "r"), delimiter=",")
        for row in reader:
            data.append(row)

        activity_data = {}
        if user in person_data:
            activity_data = person_data[user]
            activity_data[activity] = data
        else:
            activity_data[activity] = data
        person_data[user] = activity_data

    return person_data

def extract_features(data, dct_length, win_len=500):
    people = {}
    for person in data:
        person_data = data[person]
        activities = {}
        for activity in person_data:
            df = person_data[activity]
            _ws,_ts = split_windows(df, win_len, overlap_ratio=1)
            dct_ws = dct(_ws, comps=dct_length)
            dct_ts = dct(_ts, comps=dct_length)
            act = activityIdDict[activity]
            labs = [act for i in range(len(_ws))]
            activities[act] = [dct_ws, dct_ts, labs]
        people[person] = activities
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

def holdout_train_test_split(user_data, test_ids):
    train_data = {key:value for key, value in user_data.items() if key not in test_ids}
    test_data = {key:value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data

def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users)/3), False)
    test_users = [u for indd,u in enumerate(users) if indd in indices]
    return test_users

def mlp(x):
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def mlp_hat(x1, x2):
    x1 = Dense(360, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x2 = Dense(360, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x = concatenate([x1, x2])
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(classes), activation='softmax', name='mlp')(x)
    return x

def ae(x, feature_length):
    x = Dense(96, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(feature_length, name='ae')(x)
    return x

def upper(train_data, feature_length):
    upper_input = Input((feature_length,))
    upper_output = ae(upper_input)
    upper_model = Model(inputs=upper_input,outputs=upper_output)
    adam = Adam(lr=0.01)
    upper_model.compile(optimizer=adam,loss='mse')
    upper_model.fit(train_data[0], train_data[1], epochs=10, batch_size=64, verbose=0)

####################lower########################
def lower(train_data, feature_length):
    lower_input = Input((feature_length,))
    lower_output = mlp(lower_input)
    lower_model = Model(inputs=lower_input,outputs=lower_output)
    adam = Adam(lr=0.01)
    lower_model.compile(optimizer=adam,loss='categorical_crossentropy')
    lower_model.fit(train_data[0], train_data[2], epochs=10, batch_size=64, verbose=0)

##################combine######################
def combine(train_data, test_data, feature_length):
    upper_input = Input((feature_length,))
    upper_output = ae(upper_input)
    upper_model = Model(inputs=upper_input,outputs=upper_output)

    lower_input = Input((feature_length,))
    lower_output = mlp_hat(lower_input, upper_output)

    adam = Adam(lr=0.01)

    combine_model = Model(inputs=[upper_input, lower_input],outputs=[upper_output, lower_output])
    combine_model.compile(optimizer=adam,loss={'ae':'mse', 'mlp':'categorical_crossentropy'},metrics=['accuracy'])

    train_y = train_data[2]
    test_y = test_data[2]
    _train_y = np_utils.to_categorical(train_y, len(classes))
    _test_y = np_utils.to_categorical(test_y, len(classes))

    combine_model.fit([train_data[0], train_data[0]], [train_data[1], _train_y], epochs=10, batch_size=64, verbose=1)
    score = combine_model.evaluate([test_data[0], test_data[0]], [test_data[1], _test_y], verbose=1)
    print(score)

###############################################################
user_data = read_data(data_path)
test_user_ids = get_hold_out_users(list(user_data.keys()))
_train_data, _test_data = holdout_train_test_split(user_data, test_user_ids)
_lower_train_data = extract_features(_train_data)
_lower_test_data = extract_features(_test_data)
combine(_lower_train_data, _lower_test_data)

