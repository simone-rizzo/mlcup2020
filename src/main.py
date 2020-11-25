import numpy

from abGridSearchCV import abGridSearchCV
from load_data import load_monk
from network import NeuralNetwork
import matplotlib.pyplot as plt

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
topParam = { #for monk 3 0.9722
 'ALPHA': 0.8,
 'ETA': 0.1,
 'LAMBDA': 0.1,
 'activation': 'sigm',
 'epochs': 400,
 'hiddenUnits': 4
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
    'epochs': 413,
    'hidden_units': 4
}
defaultParameters['earlyStopping'] = False

total = 0
n = 100
monk = 1
trainData, trainLabels = load_monk(monk, 'train', encodeLabel=False)
testData, testLabels = load_monk(monk, 'test', encodeLabel=False)
for i in range(n):
    x = NeuralNetwork(**bestParams)
    x.fit(trainData, trainLabels, )
    testResults, testAccuracy = x.predict(testData, testLabels, acc_=True)
    # plt.plot(numpy.array(x.losses))
    # plt.plot(numpy.array(x.accuracies))
    # plt.show()
    print("accuracy test")
    print(testAccuracy)
    total += testAccuracy
print(f"final {total/n}")
