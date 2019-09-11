import numpy as np
from scipy import fftpack
import os
import csv
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import heapq
np.random.seed(1)
tf.set_random_seed(2)

window_length = 500
dct_length = 60
increment_ratio = 1
data_path = '/Users/anjanawijekoon/Data/SELFBACK/activity_data_34/merge_9/'
imus = [1, 2]

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
ids = range(len(classes))
classDict = dict(zip(classes, ids))


def write_data(results_path, data):
    if os.path.isfile(results_path):
        f = open(results_path, 'a')
        f.write(data + '\n')
    else:
        f = open(results_path, 'w')
        f.write(data + '\n')
    f.close()


def read_data(path):
    person_data = {}
    files = os.listdir(path)
    for f in [ff for ff in files if ff != '.DS_Store']:
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


def extract_features(data):
    people = {}
    for person in data:
        person_data = data[person]
        activities = {}
        for activity in person_data:
            df = person_data[activity]
            ws, ts = split_windows(df)
            act = classDict[activity]
            dct_ws = dct(ws)
            dct_ts = dct(ts)
            activities[act] = [dct_ws, dct_ts]
        people[person] = activities
    return people


def split_windows(data):
    _w = []
    _t = []
    i = 0
    N = len(data)
    increment = int(window_length * increment_ratio)
    while i + window_length < N:
        start = i
        end = start + window_length
        w = [a[:3] for a in data[start:end]]
        t = [a[3:] for a in data[start:end]]
        i = int(i + increment)
        _w.append(w)
        _t.append(t)
    return _w, _t


def dct(windows):
    dct_window = []
    for tw in windows:
        x = [t[0] for t in tw]
        y = [t[1] for t in tw]
        z = [t[2] for t in tw]

        dct_x = np.abs(fftpack.dct(x, norm='ortho'))
        dct_y = np.abs(fftpack.dct(y, norm='ortho'))
        dct_z = np.abs(fftpack.dct(z, norm='ortho'))

        v = np.array([])
        v = np.concatenate((v, dct_x[:dct_length]))
        v = np.concatenate((v, dct_y[:dct_length]))
        v = np.concatenate((v, dct_z[:dct_length]))

        dct_window.append(v)
    return dct_window


def read():
    user_data = read_data(data_path)
    feature_data = extract_features(user_data)
    return feature_data


def cos_knn(k, test_data, test_labels, train_data, train_labels):
    cosine = cosine_similarity(test_data, train_data)
    top = [(heapq.nlargest(k, range(len(i)), i.take)) for i in cosine]
    top = [[train_labels[j] for j in i[:k]] for i in top]
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)
    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == pred[j]:
            correct += 1
    return correct/float(len(test_labels))


def ed_knn(k, test_data, test_labels, train_data, train_labels):
    cosine = euclidean_distances(test_data, train_data)
    top = [(heapq.nlargest(k, range(len(i)), i.take)) for i in cosine]
    top = [[train_labels[j] for j in i[:k]] for i in top]
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)
    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == pred[j]:
            correct += 1
    return correct/float(len(test_labels))


def split(_data, _test_ids):
    train_data_ = {key: value for key, value in _data.items() if key not in _test_ids}
    test_data_ = {key: value for key, value in _data.items() if key in _test_ids}
    return train_data_, test_data_


def flatten(_data):
    flatten_w_data = []
    flatten_t_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_w_data.extend(activity_data[0])
            flatten_t_data.extend(activity_data[1])
            flatten_labels.extend([activity for i in range(len(activity_data[0]))])
    return flatten_w_data, flatten_t_data, flatten_labels

