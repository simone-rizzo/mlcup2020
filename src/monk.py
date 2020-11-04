import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

class MonkModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MonkModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def train(model, epochs):
    criterion = nn.MSELoss(reduction='sum')
    optimizer = nn.optim.SGD(model.parameters(), lr=1e-4)

    for t in range(epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pass


def test():
    pass


def read_data():
    """data = pd.read_csv("../monk/monks-1.train", header=None, delimiter=r"\s+")
    data = data.drop(axis=1, columns=7)
    dataframe_input = data.iloc[:, data.columns != 0]
    dataframe_output = data.iloc[:, 0]
    inputs = torch.tensor(dataframe_input.values, device=device)
    inputs.type(torch.LongTensor)
    out = torch.tensor(dataframe_output.values, device=device)
    return inputs, out"""
    xy = np.loadtxt('../monk/monks-1.train', dtype=str, delimiter=" ", usecols=(1, 2, 3, 4, 5, 6, 7))
    xy = xy.astype(dtype=np.float32)
    x = torch.from_numpy(xy[:, 1:])
    y = torch.from_numpy(xy[:, 0])
    return x, y


# Setting up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, y = read_data()
model = MonkModel(6, 3, 1)
print(x[0])
print(model(x[0]))
