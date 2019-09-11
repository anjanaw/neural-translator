import numpy as np
import csv
import os
from scipy import fftpack
import keras
from keras.layers import Dense, BatchNormalization, Input, Lambda, concatenate, Flatten
from keras.models import Model
import tensorflow as tf
from keras.layers.merge import _Merge
from keras.utils import plot_model
np.random.seed(123)

activityType = ["jogging", "sitting", "standing", "walk_fast", "walk_mod", "walk_slow", "upstairs", "downstairs", "lying"]
idList = range(len(activityType))
activityIdDict = dict(zip(activityType, idList))
upperActivityType = range(3)
upperIdList = [0,1,1,0,0,0,2,2,1]
upperActivityIdDict = dict(zip(idList, upperIdList))
upperActivityNameDict = dict(zip(activityType, upperIdList))
data_path = 'C:\\IdeaProjects\\Datasets\\selfback\\activity_data_50_'
#test_ids = ['007', '008', '009', '010', '011', '012', '013', '016', '018', '019']
test_ids = ['007']
lower_k_shot = 3
upper_k_shot = 9
n_upper_way = 3
n_lower_way = 9


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

def load_selfback_data(path):
    person_data = {}
    classes = os.listdir(path)
    for _class in classes:
        if os.path.isdir(os.path.join(path,_class)):
            files = os.listdir(os.path.join(path,_class))
            for f in files:
                p = f[:f.index('.')]
                if f.endswith('.csv'):
                    ff = os.path.join(path,_class)
                    ff = os.path.join(ff,f)
                    reader = csv.reader(open(ff, "r"), delimiter=",")
                    next(reader,None)
                    temp_data = []
                    for row in reader:
                        temp_data.append(row)
                    activity_data = {}
                    if p in person_data:
                        activity_data = person_data[p]
                        activity_data[_class] = temp_data
                    activity_data[_class] = temp_data
                    person_data[p] = activity_data
    return person_data

def holdout_train_test_split(user_data, test_ids):
    train_data = {key:value for key, value in user_data.items() if key not in test_ids}
    test_data = {key:value for key, value in user_data.items() if key in test_ids}
    return train_data, test_data

def extract_features(data, win_len=500):
    people = {}
    for person in data:
        person_data = data[person]
        classes = {}
        for activity in person_data:
            df = person_data[activity]
            act = activityIdDict.get(activity)
            wts = split_windows(df, win_len, overlap_ratio=1)
            dct_wts = dct(wts)
            labs = [act for i in range(len(wts))]
            if act in classes:
                classes[act][0].extend(dct_wts)
                classes[act][1].extend(labs)
            else:
                classes[act] = [dct_wts, labs]
        people[person] = classes
    return people

def split_windows(data, window_length, overlap_ratio=None):
    outputs = []
    i = 0
    N = len(data)
    increment = int(window_length * overlap_ratio)
    while i + window_length < N:
        start = i
        end = start + window_length
        outs = [a[1:] for a in data[start:end]]
        i = int(i + (increment))
        outputs.append(outs)
    return outputs

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

def packslice(lower_data_set, samples_per_class, numsamples):
    n_samples = samples_per_class * len(activityIdDict)
    support_cacheX = []
    support_upper_cacheY = []
    support_lower_cacheY = []
    target_upper_cacheY = []
    target_lower_cacheY = []

    for itr in range(numsamples):
        slice_x = np.zeros((n_samples + 1, 180))
        slice_lower_y = np.zeros((n_samples,))
        slice_upper_y = np.zeros((n_samples,))
        target_lower_y = None
        target_upper_y = None

        ind = 0
        pinds = np.random.permutation(n_samples)

        x_hat_lower_class = np.random.randint(len(activityIdDict))

        for t_index, t_class in enumerate(idList):
            data_pack = lower_data_set[t_class][0]
            example_inds = np.random.choice(len(data_pack), samples_per_class, False)

            for eind in example_inds:
                slice_x[pinds[ind], :] = data_pack[eind]
                slice_lower_y[pinds[ind]] = t_index
                slice_upper_y[pinds[ind]] = upperActivityIdDict[t_index]
                ind += 1

            if t_index == x_hat_lower_class:
                target_indx = np.random.choice(len(data_pack))
                while target_indx in example_inds:
                    target_indx = np.random.choice(len(data_pack))
                slice_x[n_samples, :] = data_pack[target_indx]
                target_lower_y = t_index
                target_upper_y = upperActivityIdDict[t_index]

        support_cacheX.append(slice_x)
        support_lower_cacheY.append(keras.utils.to_categorical(slice_lower_y, len(activityType)))
        support_upper_cacheY.append(keras.utils.to_categorical(slice_upper_y, len(upperActivityType)))
        target_upper_cacheY.append(keras.utils.to_categorical(target_upper_y, len(upperActivityType)))
        target_lower_cacheY.append(keras.utils.to_categorical(target_lower_y, len(activityType)))
    return np.array(support_cacheX), np.array(support_upper_cacheY), np.array(support_lower_cacheY), np.array(target_upper_cacheY),  np.array(target_lower_cacheY)


def create_train_instances(lower_train_data, samples_per_class, train_size):
    support_X = None; support_upper_y = None; support_lower_y = None; upper_y = None; lower_y = None
    for user_id, train_feats in lower_train_data.items():
        _support_X, _support_upper_y, _support_lower_y, _upper_y, _lower_y = packslice(train_feats, samples_per_class, train_size)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_upper_y = np.concatenate((support_upper_y, _support_upper_y))
            support_lower_y = np.concatenate((support_lower_y, _support_lower_y))
            upper_y = np.concatenate((upper_y, _upper_y))
            lower_y = np.concatenate((lower_y, _lower_y))
        else:
            support_X = _support_X
            support_upper_y = _support_upper_y
            support_lower_y = _support_lower_y
            upper_y = _upper_y
            lower_y = _lower_y

    print("Data shapes")
    print(support_X.shape)
    print(support_upper_y.shape)
    print(support_lower_y.shape)
    print(upper_y.shape)
    print(lower_y.shape)
    return [support_X, support_upper_y, support_lower_y, upper_y, lower_y]

def supportset_split(user_data, k_shot):
    support_set = {}
    test_set = {}
    for user, labels in user_data.items():
        for label, values in labels.items():
            X = values[0]
            y = values[1]
            if not user in support_set:
                support_set[user] = {label: (X[:k_shot], y[:k_shot])}
                test_set[user] = {label: (X[k_shot:], y[k_shot:])}
            else:
                support_set[user][label] = (X[:k_shot], y[:k_shot])
                test_set[user][label] = (X[k_shot:], y[k_shot:])
    return support_set, test_set

def packslice_test(lower_data_set, support_set, samples_per_class):
    n_samples = samples_per_class * len(activityIdDict)
    support_cacheX = []
    support_upper_cacheY = []
    support_lower_cacheY = []
    target_upper_cacheY = []
    target_lower_cacheY = []

    support_X = np.zeros((n_samples, 180))
    support_lower_y = np.zeros((n_samples,))
    support_upper_y = np.zeros((n_samples,))
    for i, label in enumerate(support_set.keys()):
        X = support_set[label][0]
        y = support_set[label][1]
        for j in range(len(X)):
            support_X[(i*samples_per_class)+j, :] = X[j]
            support_lower_y[(i*samples_per_class)+j] = y[j]
            support_upper_y[(i*samples_per_class)+j] = upperActivityIdDict[y[j]]


    for label in lower_data_set:
        X = lower_data_set[label][0]
        y = lower_data_set[label][1]
        for index in range(len(X)):
            slice_x = np.zeros((n_samples + 1, 180))
            slice_lower_y = np.zeros((n_samples,))
            slice_upper_y = np.zeros((n_samples,))

            slice_x[:n_samples, :] = support_X[:,:]
            slice_x[n_samples, :] = X[index]

            slice_lower_y[:n_samples] = support_lower_y[:]
            slice_upper_y[:n_samples] = support_upper_y[:]

            target_lower_y = y[index]
            target_upper_y =  upperActivityIdDict[y[index]]

            support_cacheX.append(slice_x)
            support_lower_cacheY.append(keras.utils.to_categorical(slice_lower_y, len(activityType)))
            support_upper_cacheY.append(keras.utils.to_categorical(slice_upper_y, len(upperActivityType)))
            target_lower_cacheY.append(keras.utils.to_categorical(target_lower_y, len(activityType)))
            target_upper_cacheY.append(keras.utils.to_categorical(target_upper_y, len(upperActivityType)))

    return np.array(support_cacheX), np.array(support_upper_cacheY), np.array(support_lower_cacheY), np.array(target_upper_cacheY), np.array(target_lower_cacheY)

def create_test_instance(lower_test_data, support_set, n_way, k_shot):
    support_X = None; support_upper_y = None; support_lower_y = None; upper_y = None; lower_y = None

    for user, test_feats in lower_test_data.items():
        support_data = support_set[user]
        _support_X, _support_upper_y, _support_lower_y, _upper_y, _lower_y = packslice_test(test_feats, support_data, k_shot)

        if support_X is not None:
            support_X = np.concatenate((support_X, _support_X))
            support_upper_y = np.concatenate((support_upper_y, _support_upper_y))
            support_lower_y = np.concatenate((support_lower_y, _support_lower_y))
            upper_y = np.concatenate((upper_y, _upper_y))
            lower_y = np.concatenate((lower_y, _lower_y))
        else:
            support_X = _support_X
            support_upper_y = _support_upper_y
            support_lower_y = _support_lower_y
            upper_y = _upper_y
            lower_y = _lower_y

    print("Data shapes: ")
    print(support_X.shape)
    print(support_upper_y.shape)
    print(support_lower_y.shape)
    print(upper_y.shape)
    print(lower_y.shape)
    return [support_X, support_upper_y, support_lower_y, upper_y, lower_y]

def mlp(x):
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def mlp_hat(x1, x2):
    #print(x1)
    x1 = Dense(360, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    #print(x1)
    #print(x2)
    x2 = Dense(40, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    #print(x2)
    x = concatenate([x1, x2])
    x = Dense(1200, activation='relu')(x)
    x = BatchNormalization()(x)
    return x

###################upper########################
def upper(train_data, test_data):
    upper_input = Input((n_supportset+1, 180))
    upper_inputs = []
    for index in range(n_supportset):
        upper_inputs.append(mlp(Lambda(lambda x: x[:,index,:])(upper_input)))
    upper_target = mlp(Lambda(lambda x: x[:,-1,:])(upper_input))
    upper_inputs.append(upper_target)
    upper_support_labels = Input((n_supportset,n_upper_way))
    upper_inputs.append(upper_support_labels)
    upper_attention = MatchCosine(nway=n_upper_way, n_samp=9)(upper_inputs)
    upper_model = Model(inputs=[upper_input,upper_support_labels],outputs=upper_attention)
    upper_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    upper_model.fit([train_data[0], train_data[1]], train_data[3], epochs=10, batch_size=64, verbose=0)
    score = upper_model.evaluate([test_data[0], test_data[1]], test_data[3], verbose=0)
    print(score)

####################lower########################
def lower(train_data, test_data):
    lower_input = Input((n_supportset+1, 180))
    lower_inputs = []
    for index in range(n_supportset):
        lower_inputs.append(mlp(Lambda(lambda x: x[:,index,:])(lower_input)))
    lower_target = mlp(Lambda(lambda x: x[:,-1,:])(lower_input))

    lower_inputs.append(lower_target)
    lower_support_labels = Input((n_supportset,n_lower_way))
    lower_inputs.append(lower_support_labels)
    lower_attention = MatchCosine(nway=n_lower_way, n_samp=lower_k_shot)(lower_inputs)

    lower_model = Model(inputs=[lower_input,lower_support_labels],outputs=lower_attention)
    lower_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    lower_model.fit([train_data[0], train_data[2]], train_data[4], epochs=10, batch_size=64, verbose=0)
    score = lower_model.evaluate([test_data[0], test_data[2]], test_data[4], verbose=0)
    print(score)

##################combine######################
def combine(train_data, test_data):
    upper_input = Input((n_supportset+1, 180))
    upper_inputs = []
    for index in range(n_supportset):
        upper_inputs.append(mlp(Lambda(lambda x: x[:,index,:])(upper_input)))
    upper_target = mlp(Lambda(lambda x: x[:,-1,:])(upper_input))
    upper_inputs.append(upper_target)
    upper_support_labels = Input((n_supportset,n_upper_way))
    upper_inputs.append(upper_support_labels)
    upper_attention = MatchCosine(nway=n_upper_way, n_samp=9)(upper_inputs)
    upper_model = Model(inputs=[upper_input,upper_support_labels],outputs=upper_attention)
    upper_output = upper_attention
    lower_input = Input((n_supportset+1, 180))
    lower_inputs = []

    for index in range(n_supportset):
        lower_inputs.append(mlp_hat(Lambda(lambda x: x[:,index,:])(lower_input), upper_output))
    lower_target = mlp(Lambda(lambda x: x[:,-1,:])(lower_input))
    lower_inputs.append(lower_target)
    lower_support_labels = Input((n_supportset,n_lower_way))
    lower_inputs.append(lower_support_labels)
    lower_attention = MatchCosine(nway=n_lower_way, n_samp=lower_k_shot)(lower_inputs)

    lower_model = Model(inputs=[upper_input, upper_support_labels, lower_input, lower_support_labels],outputs=[lower_attention, upper_attention])
    lower_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    lower_model.fit([train_data[0], train_data[1], train_data[0], train_data[2]], [train_data[4], train_data[3]], epochs=10, batch_size=64, verbose=0)
    score = lower_model.evaluate([test_data[0], test_data[1], test_data[0], test_data[2]], [test_data[4], test_data[3]], verbose=0)
    print(score)

##################combine######################
def combine_2(train_data, test_data):
    upper_output = Input((3,))
    lower_input = Input((n_supportset+1, 180))
    lower_inputs = []

    for index in range(n_supportset):
        lower_inputs.append(mlp_hat(Lambda(lambda x: x[:,index,:])(lower_input), upper_output))
    lower_target = mlp(Lambda(lambda x: x[:,-1,:])(lower_input))
    lower_inputs.append(lower_target)
    lower_support_labels = Input((n_supportset,n_lower_way))
    lower_inputs.append(lower_support_labels)
    lower_attention = MatchCosine(nway=n_lower_way, n_samp=lower_k_shot)(lower_inputs)

    lower_model = Model(inputs=[lower_input, upper_output, lower_support_labels],outputs=lower_attention)
    lower_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    lower_model.fit([train_data[0], train_data[3], train_data[2]], train_data[4], epochs=10, batch_size=64, verbose=1)
    score = lower_model.evaluate([test_data[0], test_data[3], test_data[2]], test_data[4], verbose=1)
    print(score)

###############################################################
user_data = load_selfback_data(data_path)
user_ids = list(user_data.keys())
for user_id in user_ids:
    print(user_id)
    _train_data, _test_data = holdout_train_test_split(user_data, [user_id])
    _lower_train_data = extract_features(_train_data)
    train_data = create_train_instances(_lower_train_data, lower_k_shot, 500)
    _lower_test_data = extract_features(_test_data)
    support_set, _lower_test_data = supportset_split(_lower_test_data, lower_k_shot)
    test_data = create_test_instance(_lower_test_data, support_set, n_lower_way, lower_k_shot)

    n_supportset = lower_k_shot * n_lower_way
    lower(train_data, test_data)

