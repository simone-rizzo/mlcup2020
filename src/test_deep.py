import numpy as np
from sklearn.model_selection import train_test_split
from load_data import load_monk
from deep_nn import DeepNeuralNetwork
import matplotlib.pyplot as plt

# defaultParameters = {
#     'layer_sizes': [17, 15, 1],
#     'ETA': 0.5,
#     'epochs': 500
# }

# monk = 2
# trainData, trainLabels = load_monk(monk, 'train', encodeLabel=False)
# testData, testLabels = load_monk(monk, 'test', encodeLabel=False)
# trainData_, validationData_, trainLabels_, validationLabels_ = train_test_split(
#         trainData, trainLabels, test_size=0.3)

# model = DeepNeuralNetwork(**defaultParameters)
# # trainData_ = np.transpose(trainData_)
# # trainLabels_ = np.transpose(trainLabels_)
# # validationData_ = np.transpose(validationData_)
# # validationLabels_ = np.transpose(validationLabels_)
# model.fit(trainData_, trainLabels_, validationData_, validationLabels_)
# val = model.feedforward(testData)
# print(val)

defaultParameters = {
    'layer_sizes': [2, 3, 1],
    'ETA': 0.5,
    'epochs': 1
}

trainData_ = np.array([[1, 1], [2,1]])
trainLabels_ = np.array([[0], [1]])
validationData_ = np.array([[1, 1]])
validationLabels_ = np.array([0])
testData = np.array([[2,2]])

model = DeepNeuralNetwork(**defaultParameters)
model.fit(trainData_, trainLabels_, validationData_, validationLabels_)
val = model.feedforward(testData)
print(val)