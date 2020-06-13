import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    '''Simple 2 layer neural net with h_units amount of hidden units, flattens the input 3 layer vector
    '''
    def __init__(self, h_units=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1900*3, h_units)
        self.fc2 = nn.Linear(h_units, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1900*3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ConvNN(nn.Module):
    '''Convolutional NN that uses 3x3 filters over the vectors'''

    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=(0, 1))
        self.conv2 = nn.Conv1d(16, 8, 5)
        self.conv3 = nn.Conv1d(8, 1, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.pool3 = nn.MaxPool1d(3, stride=3)
        self.fc1 = nn.Linear(105, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, 3, 1900)
        x = self.conv1(x)
        x = x.view(-1, 64, 1900)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x





