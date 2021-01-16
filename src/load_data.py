import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import normalize


def load_monk(file_, filetype):
    ''''''
    filename = "./data/monk/monks-{}.{}".format(file_, filetype)

    # encode function for monk data
    def encode(vector):
        two_feat = {'1': [1, 0], '2': [0, 1]}
        three_feat = {'1': [1, 0, 0], '2': [0, 1, 0], '3': [0, 0, 1]}
        four_feat = {'1': [1, 0, 0, 0], '2': [
            0, 1, 0, 0], '3': [0, 0, 1, 0], '4': [0, 0, 0, 1]}
        encode_dict = {
            '0': three_feat,
            '1': three_feat,
            '2': two_feat,
            '3': three_feat,
            '4': four_feat,
            '5': two_feat
        }
        encoded = []
        
        for i, val in enumerate(vector):
            encoded.extend(encode_dict[str(i)][str(val)])
        assert len(encoded) == 17
        return encoded

    with open(filename) as f:
        data = []
        labels = []
        for line in f.readlines():
            rows = [x for x in line.split(' ')][2:-1]
            data.append(encode(rows))
            labels.append([line[1]])

        data = np.array(data, dtype='float16')
        labels = np.array(labels, dtype='float16')

    return data, labels

def load_cup(file_name, ts_percentage=0.17):
    ''''''
    # read and preprocess data
    data = genfromtxt(file_name, delimiter=',')[:, 1:]

    # separate train and test
    ts_size = int(np.round(data.shape[0]*ts_percentage))
    step_size = int(np.round(data.shape[0]/ts_size))
    ts_data = np.zeros(shape=(ts_size, data.shape[1]))
    for i in range(ts_size):
        ts_data[i] = (data[i+step_size-1])
        data = np.delete(data, i+step_size-1, 0)

    # separate data and labels
    train_data = data[:, :-2]
    train_labels = data[:, -2:]
    test_labels = ts_data[:, -2:]
    test_data = ts_data[:, :-2]

    # train_data = normalize(train_data, axis=1, norm='l2')
    # train_labels = normalize(train_labels, axis=1, norm='l2')

    return train_data, train_labels, test_data, test_labels