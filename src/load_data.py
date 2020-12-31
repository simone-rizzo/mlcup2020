import numpy as np
import random
import operator
from numpy import linalg as LA
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def load_monk(file_, filetype, encodeLabel=False):
    filename = "./data/monk/monks-{}.{}".format(file_, filetype)

    def encode(vector, label=False):
        if label:
            twoFeatures = {'0': [1, 0], '1': [0, 1]}
            return twoFeatures[str(vector)]
        else:
            retVector = []
            twoFeatures = {'1': [1, 0], '2': [0, 1]}
            threeFeatures = {'1': [1, 0, 0], '2': [0, 1, 0], '3': [0, 0, 1]}
            fourFeatures = {'1': [1, 0, 0, 0], '2': [
                0, 1, 0, 0], '3': [0, 0, 1, 0], '4': [0, 0, 0, 1]}
            encodingDict = {
                '0': threeFeatures,
                '1': threeFeatures,
                '2': twoFeatures,
                '3': threeFeatures,
                '4': fourFeatures,
                '5': twoFeatures
            }
            for idx, val in enumerate(vector):
                retVector.extend(encodingDict[str(idx)][str(val)])
            return retVector

    with open(filename) as f:
        data_ = []
        labels = []
        for line in f.readlines():
            rows = [x for x in line.split(' ')][2:-1]
            temp = encode(rows)
            assert len(temp) == 17
            data_.append(encode(rows))
            label = line[1]
            if encodeLabel:
                label = encode(label, label=True)
            else:
                label = [label]
            labels.append(label)

        data_ = np.array(data_, dtype='float16')
        labels = np.array(labels, dtype='float16')

    return data_, labels

def load_cup(file_name,vl_percentage=0.20, ts_percentage=0.17, ):
    my_data = genfromtxt(file_name, delimiter=',') #Read the file
    my_data = my_data[:, 1:]    #Remove first column
    ts_size = int(np.round(my_data.shape[0]*ts_percentage)) #Compute the percentage of datas to take for TS, with step size
    step_size = int(np.round(my_data.shape[0]/ts_size))
    ts_data = np.zeros(shape=(ts_size, my_data.shape[1]))
    for i in range(ts_size):
        ts_data[i] = (my_data[i+step_size-1])
        my_data = np.delete(my_data, i+step_size-1, 0) #Delete ts_size datas from DS
    train_data = my_data[:, :-2] #Remove last 2 columns
    train_labels = my_data[:, -2:] #Keep only last 2 columns
    test_label = ts_data[:, -2:]
    test_data = ts_data[:, :-2]
    # train_data = normalize(train_data, axis=1, norm='l2')
    # train_labels = normalize(train_labels, axis=1, norm='l2')
    """for i in range(train_labels.shape[0]):
        riga = train_labels[i]
        riga = riga.reshape(1, -1)
        norm = LA.norm(riga, 2)"""
    # train_data_, validation_data_, train_labels_, validation_labels_ = train_test_split(train_data, train_labels, test_size = vl_percentage) #Split the TR into TR and Validation
    return train_data, train_labels, test_data, test_label