import numpy

from src.abGridSearchCV import abGridSearchCV
from src.loadData import loadMonk
from src.perceptron import perceptron
import matplotlib.pyplot as plt

trainData, trainLabels = loadMonk(3, 'train', encodeLabel=False)

defaultParameters={
    'hiddenUnits':3,
    'randomSeed':0,
    'activation':'sigm',
    'epochs':500,
    'ETA':0.4,
    'LAMBDA':0.0,
    'loss':"MEE",
    'ALPHA':0.9,
    'weightInitialization':'xav',
    'regression':False,
    'earlyStopping':True,
    'tolerance':1e-3,
    'patience':20
}

parameterGridForModelSelection={
#     'randomSeed':[0, 20],
    'hiddenUnits':[4, 16],
    'activation':['sigm', 'relu', 'tanh'],
    'ETA':[0.1, 0.2, 0.3, 0.5, 0.7,0.9],
    'LAMBDA':[0.001,0.01],
    'ALPHA':[0.2, 0.5, 0.7, 0.9]
}
# top5BestParams = abGridSearchCV(defaultParameters, parameterGridForModelSelection, trainData, trainLabels, winnerCriteria="meanLosses", validationSplit=0.3, log=False, topn=5)
# bestParams = top5BestParams[0]['params']
"""bestParams = {
 'ALPHA': 0.5,
 'ETA': 0.3,
 'LAMBDA': 0.001,
 'activation': 'sigm',
 'epochs': 413,
 'hiddenUnits': 4
}"""
bestParams = { #for monk 3 0.9722
 'ALPHA': 0.8,
 'ETA': 0.1,
 'LAMBDA': 0.1,
 'activation': 'sigm',
 'epochs': 400,
 'hiddenUnits': 4
}
defaultParameters['earlyStopping'] = False
testData, testLabels = loadMonk(3, 'test', encodeLabel=False)
x = perceptron(**defaultParameters)
x.set_params(**bestParams)
x.fit(trainData, trainLabels)
testResults, testAccuracy = x.predict(testData, testLabels, acc_=True)
print("accuracy")
print(testAccuracy)
plt.plot(numpy.array(x.losses))
plt.plot(numpy.array(x.accuracies))
plt.show()
