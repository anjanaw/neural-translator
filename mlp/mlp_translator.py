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
from keras.utils import plot_model
import pydot
import graphviz
np.random.seed(10)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
idList = range(len(classes))
activityIdDict = dict(zip(classes, idList))

data_path = 'C:/IdeaProjects/Datasets/selfback/activity_data_34_9/'

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
            activities[act] = [dct_ws, dct_ts]
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

def flatten(_data):
    w_data = []
    t_data = []
    labels = []
    for user in _data:
        activities = _data[user]
        for act in activities:
            activity = activities[act]
            w_data.extend(activity[0])
            t_data.extend(activity[1])
            labels.extend([act for i in range(len(activity[0]))])
    labels = np_utils.to_categorical(labels, len(classes))
    w_data = np.array(w_data)
    t_data = np.array(t_data)
    return [w_data, t_data, labels]


def mlp(x):
    x = Dense(600, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(classes), activation='softmax', name='mlp')(x)
    return x

def mlp_hat(x1, x2):
    x1 = Dense(600, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x2 = Dense(600, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x = concatenate([x1, x2])
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(classes), activation='softmax', name='mlp_hat')(x)
    return x

def mlp_delta(x1, x2):
    x = concatenate([x1, x2])
    x = Dense(600, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(len(classes), activation='softmax', name='mlp_delta')(x)
    return x

def ae(x):
    x = Dense(96, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(180, name='ae')(x)
    return x

def upper(train_data, test_data):
    upper_input = Input((180,))
    upper_output = ae(upper_input)
    upper_model = Model(inputs=upper_input,outputs=upper_output)
    upper_model.compile(optimizer='adam', loss='mse')
    upper_model.fit(train_data[0], train_data[1], epochs=20, batch_size=64, verbose=0)
    score = upper_model.evaluate(test_data[0], test_data[1], verbose=0)
    print(upper_model.metrics_names)
    print(score)

####################lower########################
def lower_w(train_data, test_data):
    lower_input = Input((180,))
    lower_output = mlp(lower_input)
    lower_model = Model(inputs=lower_input,outputs=lower_output)
    lower_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lower_model.fit(train_data[0], train_data[2], epochs=20, batch_size=64, verbose=0)
    score = lower_model.evaluate(test_data[0], test_data[2], verbose=0)
    print(lower_model.metrics_names)
    print(score)

def lower_t(train_data, test_data):
    lower_input = Input((180,))
    lower_output = mlp(lower_input)
    lower_model = Model(inputs=lower_input,outputs=lower_output)
    lower_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lower_model.fit(train_data[1], train_data[2], epochs=20, batch_size=64, verbose=0)
    score = lower_model.evaluate(test_data[1], test_data[2], verbose=0)
    print(lower_model.metrics_names)
    print(score)

def lower_wt(train_data, test_data):
    lower_input_1 = Input((180,))
    lower_input_2 = Input((180,))
    lower_output = mlp_hat(lower_input_1, lower_input_2)
    lower_model = Model(inputs=[lower_input_1,lower_input_2],outputs=lower_output)
    lower_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lower_model.fit([train_data[0],train_data[1]], train_data[2], epochs=20, batch_size=64, verbose=0)
    score = lower_model.evaluate([test_data[0], test_data[1]], test_data[2], verbose=0)
    print(lower_model.metrics_names)
    print(score)

##################combine######################
def combine(train_data, test_data):
    upper_input = Input((180,))
    upper_output = ae(upper_input)

    #lower_output = mlp_hat(upper_input, upper_output)
    lower_output = mlp_delta(upper_input, upper_output)

    combine_model = Model(inputs=upper_input,outputs=[upper_output, lower_output])
    #combine_model.compile(optimizer=adam,loss={'ae':'mse', 'mlp_hat':'categorical_crossentropy'},metrics=['accuracy'])
    combine_model.compile(optimizer='adam',loss={'ae':'binary_crossentropy', 'mlp_delta':'categorical_crossentropy'},metrics=['accuracy'])
    combine_model.summary()
    plot_model(combine_model, 'delta.png')

    combine_model.fit(train_data[0], [train_data[1], train_data[2]], epochs=20, batch_size=64, verbose=0)
    score = combine_model.evaluate(test_data[0], [test_data[1], test_data[2]], verbose=0)
    print(combine_model.metrics_names)
    print(score)

##################combine alpha######################
def combine_alpha(train_data, test_data):
    upper_input = Input((180,))
    upper_output = ae(upper_input)

    lower_output = mlp_hat(upper_input, upper_output)

    combine_model = Model(inputs=upper_input,outputs=[upper_output, lower_output])
    combine_model.compile(optimizer='adam',loss={'ae':'mse', 'mlp_hat':'categorical_crossentropy'},metrics=['accuracy'])
    #combine_model.summary()
    #plot_model(combine_model, 'hat.png')

    combine_model.fit(train_data[0], [train_data[1], train_data[2]], epochs=20, batch_size=64, verbose=1)
    score = combine_model.evaluate(test_data[0], [test_data[1], test_data[2]], verbose=0)
    print(combine_model.metrics_names)
    print(score)


####################segregate#################
def segregate(train_data, test_data):
    upper_input = Input((180,))
    upper_output = ae(upper_input)
    ae_model = Model(inputs=upper_input, outputs=upper_output)
    ae_model.compile(optimizer='adam', loss='mse')
    ae_model.fit(train_data[0], train_data[1], epochs=10, batch_size=64, verbose=0)
    _test_data = ae_model.predict(test_data[0])

    lower_input_1 = Input((180,))
    lower_input_2 = Input((180,))
    lower_output = mlp_hat(lower_input_1, lower_input_2)
    mlp_model = Model(inputs=[lower_input_1,lower_input_2],outputs=lower_output)
    mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mlp_model.fit([train_data[0],train_data[1]], train_data[2], epochs=20, batch_size=64, verbose=0)
    score = mlp_model.evaluate([test_data[0], _test_data], test_data[2], verbose=0)
    print(mlp_model.metrics_names)
    print(score)


###############################################################
user_data = read_data(data_path)
feature_data = extract_features(user_data, 60)
user_data = None
test_user_ids = get_hold_out_users(list(feature_data.keys()))
_train_features, _test_features = holdout_train_test_split(feature_data, test_user_ids)
_train_features = flatten(_train_features)
_test_features = flatten(_test_features)
#upper(_train_features, _test_features)
lower_w(_train_features, _test_features)
lower_t(_train_features, _test_features)
lower_wt(_train_features, _test_features)
#combine(_train_features, _test_features)
#segregate(_train_features, _test_features)
combine_alpha(_train_features, _test_features)
