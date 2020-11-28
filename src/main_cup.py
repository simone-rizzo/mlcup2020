import numpy as np
from sklearn.model_selection import train_test_split
from abGridSearchCV import abGridSearchCV
from load_data import load_cup
from src.network import NeuralNetwork
import matplotlib.pyplot as plt

def show_single_model(param, traindata, trainlabel, testdata, testlabel):
    model = NeuralNetwork(**param)
    model.fit(traindata, trainlabel)
    _, testAccuracy = model.predict(testdata, testlabel, acc_=True)
    fig, ax = plt.subplots()
    ax.plot(np.array(model.train_losses))
    fig.suptitle('Acc%: '+str(testAccuracy))
    plt.show()

def show_model_with_validation(param, train_data, train_labels,validationData_,validationLabels_, testdata, testlabel):
    model = NeuralNetwork(**param)
    model.fit(train_data, train_labels, validationData_, validationLabels_,
          early_stoppingLog=True, comingFromGridSearch=False)
    _, testAccuracy = model.predict(testdata, testlabel, acc_=True)
    fig, ax = plt.subplots()
    ax.plot(np.array(model.train_losses))
    ax.plot(np.array(model.valid_losses))
    fig.suptitle('Acc%: ' + str(testAccuracy))
    plt.show()

defaultParameters = {
    'hidden_units': 10,
    'activation': 'relu',
    'epochs': 500,
    'ETA': 0.5,
    'LAMBDA': 0.01,
    'loss': "MEE",
    'ALPHA': 0.5,
    'weight_init': 'xav',
    'regression': True,
    'tolerance': 1e-3,
    'patience': 20
}

parameterGridForModelSelection = {
    'hidden_units': [8, 12, 16, 20],
    'activation': ['sigm', 'relu', 'tanh'],
    'ETA': [0.05, 0.1, 0.2, 0.3],
    'LAMBDA': [0.001, 0.01],
    'ALPHA': [0.5, 0.6, 0.7, 0.9],
    'weight_init': ['xav']
}

labels = ['Loss', 'Val_loss']
train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_cup('../data/cup/ML-CUP20-TR.csv')
x = NeuralNetwork(**defaultParameters)
x.fit(train_data, train_labels, valid_data, valid_labels)
print(f'Loss: {x.train_losses}')
print(f'Train Accuracy: {x.train_accuracies}')
plt.plot(np.array(x.train_losses))
plt.show()
# top5BestParams = abGridSearchCV(defaultParameters, parameterGridForModelSelection, train_data, train_labels, winnerCriteria="meanValidationLoss", validationSplit=0.3, log=False, topn=9)

# fig, axs = plt.subplots(3, 3)
# r = 0
# c = 0
# for i in range(len(top5BestParams)):
#     param = (top5BestParams[i]['params'])
#     x = NeuralNetwork(**param)
#     x.fit(train_data_, train_labels_, validationData_, validationLabels_,
#               early_stoppingLog=False, comingFromGridSearch=False)
#     axs[r, c].plot(np.array(x.train_losses), label=labels[0])
#     axs[r, c].plot(np.array(x.valid_losses), label=labels[1])
#     axs[r, c].set_title("ALPHA: "+str(param['ALPHA'])+" ETA: "+str(param['ETA'])+" LAMBDA: "+str(param['LAMBDA'])+" U: "+str(param['hidden_units']))
#     axs[r, c].set(xlabel='Ephocs', ylabel='Loss')
#     axs[r, c].legend()
#     if c < 2:
#         c += 1
#     else:
#         r += 1
#         c = 0

# plt.show()
# show_single_model(bestParams, train_data, train_labels, test_data, test_labels)
# show_model_with_validation(bestParams, train_data_, train_labels_, validationData_, validationLabels_, test_data, test_labels)