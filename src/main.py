import numpy

from abGridSearchCV import abGridSearchCV
from load_data import load_monk
from network import NeuralNetwork
import matplotlib.pyplot as plt

import pathlib
print(pathlib.Path().absolute())
trainData, trainLabels = load_monk(1, 'train', encodeLabel=False)
defaultParameters = {
    'hidden_units': 3,
    'activation': 'sigm',
    'epochs': 500,
    'ETA': 0.4,
    'LAMBDA': 0.0,
    'loss': "MEE",
    'ALPHA': 0.9,
    'weightInitialization': 'xav',
    'regression': False,
    'earlyStopping': True,
    'tolerance': 1e-3,
    'patience': 20
}

parameterGridForModelSelection = {
    'hidden_units': [4, 16],
    'activation': ['sigm', 'relu', 'tanh'],
    'ETA': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    'LAMBDA': [0.001, 0.01],
    'ALPHA': [0.2, 0.5, 0.7, 0.9]
}
# top5BestParams = abGridSearchCV(defaultParameters, parameterGridForModelSelection, trainData, trainLabels, winnerCriteria="meanLosses", validationSplit=0.3, log=False, topn=5)
# bestParams = top5BestParams[0]
# print(bestParams)
bestParams = {
    'ALPHA': 0.5,
    'ETA': 0.3,
    'LAMBDA': 0.001,
    'activation': 'sigm',
    'epochs': 500,
    'hidden_units': 4
}
defaultParameters['earlyStopping'] = False

total = 0
for i in range(50):
    x = NeuralNetwork(**bestParams)
    x.fit(trainData, trainLabels, )
    testData, testLabels = load_monk(1, 'test', encodeLabel=False)
    testResults, testAccuracy = x.predict(testData, testLabels, acc_=True)
    # plt.plot(numpy.array(x.losses))
    # plt.plot(numpy.array(x.accuracies))
    # plt.show()
    print("accuracy test")
    print(testAccuracy)
    total += testAccuracy
print(f"final {total/50}")
