import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from scipy import fftpack
from sklearn.svm import SVC
import keras
import os
import csv
import tensorflow as tf
from keras.layers.merge import _Merge

imus = [1,2]

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
idList = range(len(classes))
activityIdDict = dict(zip(classes, idList))

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
    for user in _data:
        activities = _data[user]
        for act in activities:
            activity = activities[act]
            data.extend(activity)
            lbls.extend([act for i in range(len(activity))])
    return data,lbls

dct_length = 60
data_path = '/home/anjana/Datasets/selfback/activity_data_34/merge/'

user_data = read_data(data_path)
feature_data = extract_features(user_data, dct_length)
user_data = {}

for i in range(5):
    np.random.seed(i)
    test_user_ids = get_hold_out_users(list(feature_data.keys()))
    print(test_user_ids)

    _train_features, _test_features = user_holdout_split(feature_data, test_user_ids)
    _train_features, _train_labels = flatten(_train_features)
    _test_features, _test_labels = flatten(_test_features)

    svc = SVC()
    svc.fit(_train_features, _train_labels)
    score = svc.score(_test_features, _test_labels)
    print(score)