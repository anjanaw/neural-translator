import csv
import os
import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import numpy as np

frames_per_second = 1
window = 5
increment = 2

dc_frame_size = 12*16
pm_frame_size = 16*16

min_length = frames_per_second*window
max_length = 15*window

pm_path = '/Volumes/1708903/MEx/Data/pm_scaled/1.0_0.5'  # 16x16
dc_path = '/Volumes/1708903/MEx/Data/dc_scaled/0.05_0.05'  # 12x16

activity_list = ['01', '02', '03', '04', '05', '06', '07']
id_list = range(len(activity_list))
activity_id_dict = dict(zip(activity_list, id_list))


def write_data(results_file, data):
    if os.path.isfile(results_file):
        f = open(results_file, 'a')
        f.write(data + '\n')
    else:
        f = open(results_file, 'w')
        f.write(data + '\n')
    f.close()


def _read(_file):
    reader = csv.reader(open(_file, "r"), delimiter=",")
    _data = []
    for row in reader:
        if len(row[0]) == 19 and '.' not in row[0]:
            row[0] = row[0]+'.000000'
        temp = [dt.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')]
        _temp = [float(f) for f in row[1:]]
        temp.extend(_temp)
        _data.append(temp)
    return _data


def _read_(path, _sensor):
    alldata = {}
    subjects = os.listdir(path)
    for subject in subjects:
        allactivities = {}
        subject_path = os.path.join(path, subject)
        activities = os.listdir(subject_path)
        for activity in activities:
            if not activity.startswith('.'):
                sensor = activity.split('.')[0].replace(_sensor, '')
                activity_id = sensor.split('_')[0]
                sensor_index = sensor.split('_')[1]
                _data = _read(os.path.join(subject_path, activity))
                if activity_id in allactivities:
                    allactivities[activity_id][sensor_index] = _data
                else:
                    allactivities[activity_id] = {}
                    allactivities[activity_id][sensor_index] = _data
        alldata[subject] = allactivities
    return alldata


def pad(data, length):
    pad_length = []
    if length % 2 == 0:
        pad_length = [int(length / 2), int(length / 2)]
    else:
        pad_length = [int(length / 2) + 1, int(length / 2)]
    new_data = []
    for index in range(pad_length[0]):
        new_data.append(data[0])
    new_data.extend(data)
    for index in range(pad_length[1]):
        new_data.append(data[len(data) - 1])
    return new_data


def reduce(data, length):
    red_length = []
    if length % 2 == 0:
        red_length = [int(length / 2), int(length / 2)]
    else:
        red_length = [int(length / 2) + 1, int(length / 2)]
    new_data = data[red_length[0]:len(data) - red_length[1]]
    return new_data


def pad_features(_features):
    new_features = {}
    for subject in _features:
        new_activities = {}
        activities = _features[subject]
        for act in activities:
            items = activities[act]
            new_items = []
            for item in items:
                dc_len = len(item[0])
                pm_len = len(item[1])
                new_item = []
                if dc_len < min_length or pm_len < min_length:
                    continue
                if dc_len > max_length:
                    new_item.append(reduce(item[0], dc_len - max_length))
                elif dc_len < max_length:
                    new_item.append(pad(item[0], max_length - dc_len))
                else:
                    new_item.append(item[0])

                if pm_len > max_length:
                    new_item.append(reduce(item[1], pm_len - max_length))
                elif pm_len < max_length:
                    new_item.append(pad(item[1], max_length - pm_len))
                else:
                    new_item.append(item[1])
                new_items.append(new_item)
            new_activities[act] = new_items
        new_features[subject] = new_activities
    return new_features


def find_index(_data, _time_stamp):
    return [_index for _index, _item in enumerate(_data) if _item[0] >= _time_stamp][0]


def trim(_data):
    _length = len(_data)
    _inc = _length/(window*frames_per_second)
    _new_data = []
    for i in range(window*frames_per_second):
        _new_data.append(_data[i*_inc])
    return _new_data


def frame_reduce(_data):
    if frames_per_second == 0:
        return _data
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            time_windows = []
            for item in activity_data:
                new_item = []
                new_item.append(trim(item[0]))
                new_item.append(trim(item[1]))
                time_windows.append(new_item)
            _activities[activity] = time_windows
        _features[subject] = _activities
    return _features


def split_windows(dc_data, pm_data):
    outputs = []
    start = dc_data[0][0]
    end = dc_data[len(dc_data) - 1][0]
    _increment = dt.timedelta(seconds=increment)
    _window = dt.timedelta(seconds=window)

    dc_frames = [a[1:] for a in dc_data[:]]

    pm_frames = [a[1:] for a in pm_data[:]]
    pm_frames = np.array(pm_frames)
    _length = pm_frames.shape[0]
    pm_frames = np.reshape(pm_frames, (_length*pm_frame_size))
    pm_frames = pm_frames/max(pm_frames)
    pm_frames = [float("{0:.5f}".format(f)) for f in pm_frames.tolist()]
    pm_frames = np.reshape(np.array(pm_frames), (_length, pm_frame_size))

    while start + _window < end:
        _end = start + _window
        dc_start_index = find_index(dc_data, start)
        dc_end_index = find_index(dc_data, _end)
        pm_start_index = find_index(pm_data, start)
        pm_end_index = find_index(pm_data, _end)
        dc_instances = [a[:] for a in dc_frames[dc_start_index:dc_end_index]]
        pm_instances = [a[:] for a in pm_frames[pm_start_index:pm_end_index]]
        start = start + _increment
        instances = [dc_instances, pm_instances]
        outputs.append(instances)
    return outputs


def extract_features(dc_data, pm_data):
    _features = {}
    for subject in dc_data:
        _activities = {}
        dc_activities = dc_data[subject]
        for dc_activity in dc_activities:
            time_windows = []
            activity_id = activity_id_dict.get(dc_activity)
            dc_activity_data = dc_activities[dc_activity]
            pm_activity_data = pm_data[subject][dc_activity]
            for item in dc_activity_data.keys():
                time_windows.extend(split_windows(dc_activity_data[item], pm_activity_data[item]))
            _activities[activity_id] = time_windows
        _features[subject] = _activities
    return _features


def merge(_data):
    _features = {}
    for subject in _data:
        _activities = {}
        activities = _data[subject]
        for activity in activities:
            _items = []
            items = activities[activity]
            for item in items:
                timestamps = []
                for i in range(len(item[0])):
                    timestamp = [a[i] for a in item]
                    dc = timestamp[0].tolist()
                    pm = timestamp[1].tolist()
                    dc.extend(pm)
                    timestamps.append(dc)
                _items.append(timestamps)
            _activities[activity] = _items
        _features[subject] = _activities
    return _features


def read():
    dc_data = _read_(dc_path, '_dc')
    pm_data = _read_(pm_path, '_pm')
    all_features = extract_features(dc_data, pm_data)
    all_features = pad_features(all_features)
    all_features = frame_reduce(all_features)
    return all_features


def split(_data, test_ids):
    train_data_ = {key: value for key, value in _data.items() if key not in test_ids}
    test_data_ = {key: value for key, value in _data.items() if key in test_ids}
    return train_data_, test_data_


def flatten(_data):
    flatten_dc_data = []
    flatten_pm_data = []
    flatten_labels = []

    for subject in _data:
        activities = _data[subject]
        for activity in activities:
            activity_data = activities[activity]
            flatten_dc_data.extend([a[0] for a in activity_data])
            flatten_pm_data.extend([a[1] for a in activity_data])
            flatten_labels.extend([activity for i in range(len(activity_data))])

    return flatten_dc_data, flatten_pm_data, flatten_labels


def cos_knn(k, test_data, test_labels, train_data, train_labels):
    cosim = cosine_similarity(test_data, train_data)

    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    top = [[train_labels[j] for j in i[:k]] for i in top]

    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)

    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == pred[j]:
            correct += 1
    acc = correct / float(len(test_labels))
    return acc
