import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from scipy import fftpack
import os
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
imus = [1,2]

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
idList = range(len(classes))
activityIdDict = dict(zip(classes, idList))

def write_data(data):
    file_path = 'C:\\IdeaProjects\\Datasets\\selfback\\34_9_3d_data.csv'
    if(os.path.isfile(file_path)):
        f = open(file_path, 'a')
        f.write(data+'\n')
    else:
        f = open(file_path, 'w')
        f.write(data+'\n')
    f.close()

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
            _wts = split_windows(df, win_len, overlap_ratio=1)
            dct_wts = dct(_wts, comps=dct_length)
            act = activityIdDict[activity]
            activities[act] = dct_wts
        people[person] = activities
    return people

def split_windows(data, window_length, overlap_ratio=None):
    wt = []
    i = 0
    N = len(data)
    increment = int(window_length * overlap_ratio)
    while i + window_length < N:
        start = i
        end = start + window_length
        _wt = [a[:] for a in data[start:end]]
        i = int(i + (increment))
        wt.append(_wt)
    return wt

def dct(windows, comps=60):
    dct_window = []
    for tw in windows:
        all_acc_dcts = np.array([])
        for index in imus:
            _index = index - 1
            x = [t[(_index*3)+0] for t in tw]
            y = [t[(_index*3)+1] for t in tw]
            z = [t[(_index*3)+2] for t in tw]

            dct_x = np.abs(fftpack.dct(x, norm='ortho'))
            dct_y = np.abs(fftpack.dct(y, norm='ortho'))
            dct_z = np.abs(fftpack.dct(z, norm='ortho'))

            v = np.array([])
            v = np.concatenate((v, dct_x[:comps]))
            v = np.concatenate((v, dct_y[:comps]))
            v = np.concatenate((v, dct_z[:comps]))
            all_acc_dcts = np.concatenate((all_acc_dcts, v))

        dct_window.append(all_acc_dcts)
    return dct_window

def mlp():
    x_input = Input((360,))
    x = Dense(1200, activation='relu')(x_input)
    x = BatchNormalization()(x)
    x = Dense(600, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(3, activation='relu')(x)
    x = BatchNormalization()(x)
    x_output = Dense(9, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x_output)
    model.summary()
    return model

def user_holdout_split(user_data, test_ids):
    train_data = {key:value for key, value in user_data.items() if key not in test_ids}
    test_data = {key:value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data

def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users)/3), False)
    test_users = [u for indd,u in enumerate(users) if indd in indices]
    return test_users

def flatten(_data):
    data = []
    lbls = []
    users = []
    for user in _data:
        activities = _data[user]
        for act in activities:
            activity = activities[act]
            data.extend(activity)
            lbls.extend([act for i in range(len(activity))])
            users.extend([user for j in range(len(activity))])
    return data,lbls,users

def run_model():
    dct_length = 60
    data_path = 'C:\\IdeaProjects\\Datasets\\selfback\\activity_data_34_9\\'

    user_data = read_data(data_path)
    feature_data = extract_features(user_data, dct_length)
    _features, _labels, _users = flatten(feature_data)
    _features = np.array(_features, dtype=np.dtype('Float64'))
    print(_features.shape)
    _labells = np_utils.to_categorical(_labels, 9)
    print(_labells.shape)
    model = mlp()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(_features, _labells, epochs=20, verbose=1, shuffle=True)

    model_output = model.layers[6].output
    model_encoder = K.function([model.input, K.learning_phase()], [model_output])
    target_embeds = model_encoder([_features, 0])[0]
    target_embeds = np.array(target_embeds, dtype=np.dtype('Float64'))
    for item,_l,_u in zip(target_embeds.tolist(), _labels, _users):
        write_data(','.join([str(t) for t in item])+','+str(_l)+','+str(_u))

def _plot():
    file_path = 'C:\\IdeaProjects\\Datasets\\selfback\\34_9_3d_data.csv'
    data_dict = {}
    user_dict = {}
    reader = csv.reader(open(file_path, "r"), delimiter=",")
    for row in reader:
        x = float(row[0])
        y = float(row[1])
        z = float(row[2])
        l = int(row[3])
        u = str(row[4])
        if x<10 and y < 10 and z< 10:
            if l in data_dict:
                _x, _y, _z = data_dict[l]
                _x.append(x)
                _y.append(y)
                _z.append(z)
                data_dict[l] = [_x, _y, _z]
            else:
                _x = [x]
                _y = [y]
                _z = [z]
                data_dict[l] = [_x, _y, _z]

        x = float(row[0])
        y = float(row[1])
        z = float(row[2])
        u = str(row[4])
        if x<10 and y < 10 and z< 10:
            if u in user_dict:
                _x, _y, _z = user_dict[u]
                _x.append(x)
                _y.append(y)
                _z.append(z)
                user_dict[u] = [_x, _y, _z]
            else:
                _x = [x]
                _y = [y]
                _z = [z]
                user_dict[u] = [_x, _y, _z]

    fig = plt.figure()

    ax = Axes3D(fig)
    _value = user_dict['026']
    ax.scatter(_value[0], _value[1], _value[2], c='b', marker=',')

    _value = user_dict['027']
    ax.scatter(_value[0], _value[1], _value[2], c='g', marker='v')

    _value = user_dict['028']
    ax.scatter(_value[0], _value[1], _value[2], c='r', marker='^')

    _value = user_dict['029']
    ax.scatter(_value[0], _value[1], _value[2], c='c', marker='<')

    #_value = user_dict['030']
    #ax.scatter(_value[0], _value[1], _value[2], c='m', marker='>')

    #_value = user_dict['031']
    #ax.scatter(_value[0], _value[1], _value[2], c='y', marker='8')

    #_value = user_dict['036']
    #ax.scatter(_value[0], _value[1], _value[2], c='k', marker='s')

    #_value = user_dict['033']
    #ax.scatter(_value[0], _value[1], _value[2], c='burlywood', marker='*')

    #_value = user_dict['034']
    #ax.scatter(_value[0], _value[1], _value[2], c='chartreuse', marker='+')
    plt.show()


_plot()


'''
    fig = plt.figure()
    ax = Axes3D(fig)
    _value = data_dict[0]
    ax.scatter(_value[0], _value[1], _value[2], c='b', marker=',')

    _value = data_dict[1]
    ax.scatter(_value[0], _value[1], _value[2], c='g', marker='v')

    _value = data_dict[2]
    ax.scatter(_value[0], _value[1], _value[2], c='r', marker='^')

    _value = data_dict[3]
    ax.scatter(_value[0], _value[1], _value[2], c='c', marker='<')

    _value = data_dict[4]
    ax.scatter(_value[0], _value[1], _value[2], c='m', marker='>')

    _value = data_dict[5]
    ax.scatter(_value[0], _value[1], _value[2], c='y', marker='8')

    _value = data_dict[6]
    ax.scatter(_value[0], _value[1], _value[2], c='k', marker='s')

    _value = data_dict[7]
    ax.scatter(_value[0], _value[1], _value[2], c='burlywood', marker='*')

    _value = data_dict[8]
    ax.scatter(_value[0], _value[1], _value[2], c='chartreuse', marker='+')
    plt.show()
'''

