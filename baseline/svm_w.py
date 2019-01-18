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

imus = [2]

classes = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
idList = range(len(classes))
activityIdDict = dict(zip(classes, idList))

def write_data(file_path, data):
    if(os.path.isfile(file_path)):
        f = open(file_path, 'a')
        f.write(data+'\n')
    else:
        f = open(file_path, 'w')
        f.write(data+'\n')
    f.close()

class MatchCosine(_Merge):
    def __init__(self,nway=5, n_samp=1, **kwargs):
        super(MatchCosine,self).__init__(**kwargs)
        self.eps = 1e-10
        self.nway = nway
        self.n_samp = n_samp

    def build(self, input_shape):
        print('here')

    def call(self,inputs):
        self.nway = (len(inputs)-2)/self.n_samp
        similarities = []

        targetembedding = inputs[-2]
        numsupportset = len(inputs)-2
        for ii in range(numsupportset):
            supportembedding = inputs[ii]

            sum_support = tf.reduce_sum(tf.square(supportembedding), 1, keep_dims=True)
            supportmagnitude = tf.rsqrt(tf.clip_by_value(sum_support, self.eps, float("inf")))

            sum_query = tf.reduce_sum(tf.square(targetembedding), 1, keep_dims=True)
            querymagnitude = tf.rsqrt(tf.clip_by_value(sum_query, self.eps, float("inf")))

            dot_product = tf.matmul(tf.expand_dims(targetembedding,1),tf.expand_dims(supportembedding,2))
            dot_product = tf.squeeze(dot_product,[1])

            cosine_similarity = dot_product*supportmagnitude*querymagnitude
            similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1,values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities,1),inputs[-1]))

        preds.set_shape((inputs[0].shape[0],self.nway ))
        return preds

    def compute_output_shape(self,input_shape):
        input_shapes = input_shape
        return (input_shapes[0][0],self.nway )

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
            dct_ts = dct(_ts, comps=dct_length)
            act = activityIdDict[activity]
            activities[act] = dct_ts
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

def support_set_split(_data, k_shot):
    support_set = {}
    everything_else = {}
    for user, labels in _data.items():
        _support_set = {}
        _everything_else = {}
        for label, data in labels.items():
            supportset_indexes = np.random.choice(range(len(data)), k_shot, False)
            supportset = [d for index, d in enumerate(data) if index in supportset_indexes]
            everythingelse = [d for index, d in enumerate(data) if index not in supportset_indexes]
            _support_set[label] = supportset
            _everything_else[label] = everythingelse
        support_set[user] = _support_set
        everything_else[user] = _everything_else
    return support_set, everything_else

def packslice(data_set, classes_per_set, samples_per_class, train_size, feature_length):
    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []

    for itr in range(train_size):
        slice_x = np.zeros((n_samples + 1, feature_length))
        slice_y = np.zeros((n_samples,))
        ind = 0
        pinds = np.random.permutation(n_samples)
        slice_classes = np.random.choice(list(data_set.keys()), classes_per_set, False)
        x_hat_class = np.random.randint(classes_per_set)

        for j, cur_class in enumerate(slice_classes):
            data_pack = data_set[cur_class]
            example_inds = np.random.choice(len(data_pack), samples_per_class, False)

            for eind in example_inds:
                slice_x[pinds[ind], :] = data_pack[eind]
                slice_y[pinds[ind]] = cur_class
                ind += 1

            if j == x_hat_class:
                target_indx = np.random.choice(len(data_pack))
                while target_indx in example_inds:
                    target_indx = np.random.choice(len(data_pack))
                slice_x[n_samples, :] = data_pack[target_indx]
                target_y = cur_class

        support_cacheX.append(slice_x)
        support_cacheY.append(keras.utils.to_categorical(slice_y, classes_per_set))
        target_cacheY.append(keras.utils.to_categorical(target_y, classes_per_set))

    return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)

def create_train_instances(train_sets, classes_per_set, samples_per_class, train_size, feature_length):
    support_X = None; support_y = None; target_y = None
    for user_id, train_feats in train_sets.items():
        _support_X, _support_y, _target_y = packslice(train_feats, classes_per_set, samples_per_class, train_size, feature_length)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_y = np.concatenate((support_y, _support_y))
            target_y = np.concatenate((target_y, _target_y))
        else:
            support_X = _support_X
            support_y = _support_y
            target_y = _target_y

    print("Data shapes: ")
    print(support_X.shape)
    print(support_y.shape)
    print(target_y.shape)
    return [support_X, support_y, target_y]
#
def packslice_test(data_set, support_set, classes_per_set, samples_per_class, feature_length):
    n_samples = samples_per_class * classes_per_set
    support_cacheX = []
    support_cacheY = []
    target_cacheY = []

    support_X = np.zeros((n_samples, feature_length))
    support_y = np.zeros((n_samples,))
    for i, _class in enumerate(support_set.keys()):
        X = support_set[_class]
        for j in range(len(X)):
            support_X[(i*samples_per_class)+j, :] = X[j]
            support_y[(i*samples_per_class)+j] = _class

    for _class in data_set:
        X = data_set[_class]
        for itr in range(len(X)):
            slice_x = np.zeros((n_samples + 1, feature_length))
            slice_y = np.zeros((n_samples,))

            slice_x[:n_samples, :] = support_X[:]
            slice_x[n_samples, :] = X[itr]

            slice_y[:n_samples] = support_y[:]

            target_y = _class

            support_cacheX.append(slice_x)
            support_cacheY.append(keras.utils.to_categorical(slice_y, classes_per_set))
            target_cacheY.append(keras.utils.to_categorical(target_y, classes_per_set))

    return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)

def create_test_instance(test_set, support_set, classes_per_set, samples_per_class, feature_length):
    support_X = None; support_y = None; target_y = None

    for user_id, test_data in test_set.items():
        support_data = support_set[user_id]
        _support_X, _support_y, _target_y = packslice_test(test_data, support_data, classes_per_set, samples_per_class, feature_length)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_y = np.concatenate((support_y, _support_y))
            target_y = np.concatenate((target_y, _target_y))
        else:
            support_X = _support_X
            support_y = _support_y
            target_y = _target_y

    print("Data shapes: ")
    print(support_X.shape)
    print(support_y.shape)
    print(target_y.shape)
    return [support_X, support_y, target_y]

def mlp_embedding(x):
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def user_holdout_split(user_data, test_ids):
    train_data = {key:value for key, value in user_data.items() if key not in test_ids}
    test_data = {key:value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data

def get_hold_out_users(users):
    indices = np.random.choice(len(users), int(len(users)/3), False)
    test_users = [u for indd,u in enumerate(users) if indd in indices]
    return test_users

def get_test_classes(all_clsses, test_lngth):
    indices = np.random.choice(len(all_clsses), test_lngth, False)
    test_clss = [u for indd,u in enumerate(all_clsses) if indd in indices]
    return test_clss

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

bsize = 128
samples_per_class = 5
train_size = 500
epochs =10
dct_length = 60
feature_length = dct_length * 3 * len(imus)
classes_per_set = len(classes)
data_path = 'C:\\IdeaProjects\\Datasets\\selfback\\activity_data_34_9\\'

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
