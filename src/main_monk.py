from sklearn.model_selection import train_test_split
from grid_search import model_selection, model_assessment
from load_data import load_monk

params_grid = {
    'layer_sizes': [[17, 4, 1]],
    'ETA': [0.4, 0.5, 0.6],
    'LAMBDA': [0.01, 0.05, 0.1],
    'ALPHA': [0.9, 0.99],
}

monk = 2
train_data, train_labels = load_monk(monk, 'train', encodeLabel=False)
test_data, test_labels = load_monk(monk, 'test', encodeLabel=False)
train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.3)

best_params = model_selection(params_grid, train_data, train_labels, valid_data, valid_labels)
model_assessment(best_params[0]['params'], train_data, train_labels, valid_data, valid_labels, test_data, test_labels)
print(best_params)
# model = DeepNeuralNetwork(**params)
# model.fit(train_data, train_labels, valid_data, valid_labels)
# plt.plot(model.train_losses)
# plt.plot(model.valid_losses)
# plt.show()