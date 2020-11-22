import numpy

from src.abGridSearchCV import abGridSearchCV
from src.loadData import loadMonk
from src.perceptron import perceptron
import matplotlib.pyplot as plt

trainData, trainLabels = loadMonk(1, 'train', encodeLabel=False)

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
bestParams = {
'ALPHA': 0.5,
 'ETA': 0.3,
 'LAMBDA': 0.001,
 'activation': 'sigm',
 'epochs': 413,
 'hiddenUnits': 4
}
defaultParameters['earlyStopping'] = False
x = perceptron(**defaultParameters)
x.set_params(**bestParams)
x.fit(trainData, trainLabels)
testData, testLabels = loadMonk(1, 'test', encodeLabel=False)
testResults, testAccuracy = x.predict(testData, testLabels, acc_=True)
plt.plot(numpy.array(x.losses))
plt.plot(numpy.array(x.accuracies))
plt.show()
print("accuracy")
print(testAccuracy)
