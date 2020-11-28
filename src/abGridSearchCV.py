from sklearn.model_selection import train_test_split
from itertools import product
from network import NeuralNetwork
import numpy as np


def abGridSearchCV(regression, paramGrid, trainData_, trainLabels_, validationData_, validationLabels_, winnerCriteria, log=True, topn=5, logToBeReturned=[]):
    """
    Returns the log(if log=True) and best parameters for the 'NeuralNetwork' model by evaluating the model on all the combinations of given parameters.
    Usage:

    logOfGridSearch, topParams=abGridSearchCV(defaultParams, paramGrid, features, labels, validationSplit, winnerCriteria, topn)

    defaultParams: dict of default parameters. (This doesn't affect the model selection)
    paramGrid: dict of parameters as keys with list of params to be tested as values.
    features:features
    labels:labels
    validationSplit:ratio of validation set in overall dataset provided
    winnerCriteria: 'meanTrainingLoss' for the minimum training loss overall, 'meanValidationLoss' with minimum mean training with validation loss, meanLosses for mean of meanLoss and meanValidationLoss'
    log:if True, returns the log of all the combinations,
    topn:how many winners
    """

    assert winnerCriteria in ["meanLosses", "meanTrainingLoss",
                              "meanValidationLoss"], "This function currently doesn't support the winner criteria provided. Please make sure it's 'meanLosses' or 'meanTrainingLoss', or 'meanValidationLoss'"

    listParams = list(paramGrid.values())
    winners = []
    allCombinations = list(product(*listParams))
    for index, val in enumerate(allCombinations):
        print("                        {}/{}".format(index,
                                                     len(allCombinations)), end="\r")
        param = {}
        for index_ in range(len(paramGrid)):
            param[list(paramGrid.keys())[index_]] = val[index_]
        model = NeuralNetwork(**param)
        model.regression = False
        model.fit(trainData_, trainLabels_, validationData_, validationLabels_,
                  early_stoppingLog=False, comingFromGridSearch=True)
        
        meanLoss = np.mean(model.train_losses)
        meanValidationLoss = np.mean(model.valid_losses)
        # if model.newEpochNotification:
        # param['epochs']=model.bestEpoch
        tempLog = {
            'params': param,
            'meanTrainingLoss': meanLoss,
            'meanValidationLoss': meanValidationLoss,
            'meanLosses': (meanLoss+meanValidationLoss)/2
        }
        if log:
            logToBeReturned.append(tempLog)

        if len(winners) < topn:
            winners.append(tempLog)
        else:
            winners = sorted(winners, key=lambda k: k[winnerCriteria])
            if tempLog[winnerCriteria] < winners[-1][winnerCriteria]:
                winners[-1] = tempLog

    if log:
        return logToBeReturned, winners
    else:
        return winners
