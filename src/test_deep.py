import numpy as np
from sklearn.model_selection import train_test_split
from src.load_data import load_monk
from src.deep_nn import DeepNeuralNetwork
import matplotlib.pyplot as plt

defaultParameters = {
     'layer_sizes': [17, 15, 1],
     'ETA': 0.5,
     'epochs': 500
 }

monk = 2
trainData, trainLabels = load_monk(monk, 'train', encodeLabel=False)
testData, testLabels = load_monk(monk, 'test', encodeLabel=False)
trainData_, validationData_, trainLabels_, validationLabels_ = train_test_split(
         trainData, trainLabels, test_size=0.3)

model = DeepNeuralNetwork(**defaultParameters)
trainData_ = np.transpose(trainData_)
trainLabels_ = np.transpose(trainLabels_)
validationData_ = np.transpose(validationData_)
validationLabels_ = np.transpose(validationLabels_)
model.fit(trainData_, trainLabels_, validationData_, validationLabels_)
val = model.feedforward(testData)
print(val)

defaultParameters = {
    'layer_sizes': [2, 3, 1],
    'ETA': 0.5,
    'epochs': 1
}

"""np.random.seed(0)
train_data = np.array([[1, 1], [2, 1]])
train_labels = np.array([[0], [1]])
test_data = np.array([[2, 2]])

model = DeepNeuralNetwork(**defaultParameters)
model.fit(train_data, train_labels, train_data, train_labels)
val = model.feedforward(test_data)
print(val)"""


# defaultParameters = {
#     'layer_sizes': [1, 100, 1],
#     'ETA': 0.5,
#     'epochs': 1,
#     'act_out': 'relu'
# }
# X = 2 * np.pi * np.random.rand(100).reshape(-1, 1)
# y = np.sin(X)
# nn = DeepNeuralNetwork(**defaultParameters)
# nn.fit(X, y, X, y)
# # nn.feed_forward(X)
# print(nn.feedforward(X))