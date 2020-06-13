import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    '''Simple 2 layer neural net with h_units amount of hidden units, flattens the input 3 layer vector
    '''
    def __init__(self, h_units=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1900*3, h_units)
        self.fc2 = nn.Linear(h_units, 1)

    def forward(self, x):
        x = x.view(-1, 1900*3)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
