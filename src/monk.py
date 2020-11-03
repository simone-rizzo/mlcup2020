import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class MonkModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MonkModel, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


def train(model, epochs):
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

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
    data = pd.read_csv("monk/monks-1.train", header=None, delimiter=r"\s+")
    data = data.drop(axis=1, columns=7)
    return data 


print(read_data())