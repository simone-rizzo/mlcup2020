import numpy as np
from sklearn.model_selection import train_test_split
from abGridSearchCV import abGridSearchCV
from load_data import load_monk
from network import NeuralNetwork
import matplotlib.pyplot as plt



def show_single_model(param, traindata, trainlabel, testdata, testlabel):
    model = NeuralNetwork(**param)
    model.regression = False
    model.fit(traindata, trainlabel)
    _, testAccuracy = model.predict(testdata, testlabel, acc_=True)
    fig, ax = plt.subplots()
    ax.plot(np.array(model.train_losses))
    fig.suptitle('Acc%: '+str(testAccuracy))
    plt.show()

def show_model_with_validation(param, trainData, trainLabels,validationData_,validationLabels_, testdata, testlabel):
    model = NeuralNetwork(**param)
    model.fit(trainData, trainLabels, validationData_, validationLabels_,
          early_stoppingLog=True, comingFromGridSearch=False)
    _, testAccuracy = model.predict(testdata, testlabel, acc_=True)
    fig, ax = plt.subplots()
    ax.plot(np.array(model.train_losses))
    ax.plot(np.array(model.valid_losses))
    fig.suptitle('Acc%: ' + str(testAccuracy))
    plt.show()

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
 'hidden_units': 4
}
parameterGridForModelSelection = {
    'hidden_units': [4],
    'activation': ['sigm'],
    'ETA': [0.05, 0.1, 0.2, 0.3],
    'LAMBDA': [0.001, 0.01],
    'ALPHA': [0.5, 0.6, 0.7, 0.9]
}

bestParams = { #97.45% on monk 3
    'ALPHA': 0.5,
    'ETA': 0.1,
    'LAMBDA': 0.001,
    'activation': 'sigm',
    'epochs': 413,
    'hidden_units': 4
}
monk = 3
labels = ['Loss', 'Val_loss']
defaultParameters['earlyStopping'] = False
trainData, trainLabels = load_monk(monk, 'train', encodeLabel=False)
testData, testLabels = load_monk(monk, 'test', encodeLabel=False)
trainData_, validationData_, trainLabels_, validationLabels_ = train_test_split(
        trainData, trainLabels, test_size=0.3)
top5BestParams = abGridSearchCV(False, parameterGridForModelSelection, trainData_, trainLabels_, validationData_, validationLabels_, winnerCriteria="meanValidationLoss", log=False, topn=9)
fig, axs = plt.subplots(3, 3)
r = 0
c = 0

for i in range(len(top5BestParams)):
    param = (top5BestParams[i]['params'])
    x = NeuralNetwork(**param)
    x.fit(trainData_, trainLabels_, validationData_, validationLabels_,
              early_stoppingLog=False, comingFromGridSearch=False)
    testResults, testAccuracy = x.predict(testData, testLabels, acc_=True)
    axs[r, c].plot(np.array(x.train_losses), label=labels[0])
    axs[r, c].plot(np.array(x.valid_losses), label=labels[1])
    axs[r, c].set_title("ALPHA: "+str(param['ALPHA'])+" ETA: "+str(param['ETA'])+" LAMBDA: "+str(param['LAMBDA'])+" U: "+str(param['hidden_units']))
    axs[r, c].set(xlabel='Ephocs', ylabel='Loss')
    axs[r, c].legend()
    print(str(i)+" accuracy: "+str(testAccuracy))
    if c < 2:
        c += 1
    else:
        r += 1
        c = 0

plt.show()
# show_single_model(bestParams, trainData, trainLabels, testData, testLabels)

"""
x = [1, 2, 3, 4]
x2 = [3, 3, 3, 3]

fig, axs = plt.subplots(1, 2)
axs[0].plot(x2, label=labels[1])
axs[0].plot(x, label=labels[0])

axs[0].set_title(str(bestParams))
axs[1].plot(x2, label=labels[1])
axs[1].plot(x, label=labels[0])
plt.legend()
plt.show()
# show_model_with_validation(bestParams, trainData_, trainLabels_, validationData_, validationLabels_, testData, testLabels)
"""