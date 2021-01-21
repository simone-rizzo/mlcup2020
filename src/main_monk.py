from network import DeepNeuralNetwork
from network_utils import ensemble_assessment, model_selection, model_assessment, plot_models
from load_data import load_monk
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

params_grid = {
    'layer_sizes': [[17, 4, 1]],
    'ETA': list(np.linspace(0.1, 0.9, 9)),
    'LAMBDA': [0],
    'ALPHA': list(np.linspace(0.1, 0.9, 9)),
    'weight_init': ['monk'],
    'act_hidden': ['relu', 'sigm', 'tanh'],
    'epochs': [400],
}

# params_grid = {
#     'layer_sizes': [[17, 4, 1]],
#     'ETA': list(np.linspace(0.1, 0.5, 5)),
#     'LAMBDA': [0],
#     'ALPHA': list(np.linspace(0.5, 0.9, 5)),
#     'weight_init': ['monk'],
#     'act_hidden': ['sigm', 'tanh', 'relu'],
#     'epochs': [1000],
# }

monk = 1
train_data, train_labels = load_monk(monk, 'train')
test_data, test_labels = load_monk(monk, 'test')

best_params = model_selection(params_grid, train_data, train_labels, topn=9, kfold=4)

# train the best model
train_data, valid_data, train_labels, valid_labels = train_test_split(
    train_data, train_labels, test_size=0.2)
best_model = DeepNeuralNetwork(**best_params[0]['params'])
best_model.fit(train_data, train_labels, valid_data, valid_labels)

# best_model = model_assessment(best_params, train_data, train_labels, test_data, test_labels)
# best_model = ensemble_assessment(best_params, train_data, train_labels, test_data, test_labels)
# print(f'Best model parameters { best_params[0] }')
# print(f'Best model parameters { best_params }')

# uncomment if want to save the result
f = open(f"monk-{ monk }-configuration.txt", "w")
f.write('\n'.join([str(param) for param in best_params]))
f.write('\n')
f.write(str(np.mean([float(param['valid_loss']) for param in best_params])))

# plot and save models
_, axs = plt.subplots(1, 2)
axs[0].plot(np.array(best_model.train_losses), 'b-', label='Train Loss')
axs[0].plot(np.array(best_model.valid_losses), 'r--', label='Valid Loss')
axs[0].set(xlabel='Ephocs', ylabel='Loss')
axs[0].legend()

axs[1].plot(np.array(best_model.train_accuracies), 'b-', label='Train Accuracy')
axs[1].plot(np.array(best_model.valid_accuracies), 'r--', label='Valid Accuracy')
axs[1].set(xlabel='Ephocs', ylabel='Accuracy')
axs[1].legend()
plt.gcf().set_size_inches((30, 10), forward=False)
plt.savefig(f"./plots/monk-{ monk }.pdf", bbox_inches='tight')
plt.show()