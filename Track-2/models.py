import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    '''Simple 2 layer neural net with h_units amount of hidden units, flattens the input 3 layer vector
    '''
    def __init__(self, h_units=64, fusion=True, dropout=0.5):
        super(SimpleNN, self).__init__()
        self.fusion = fusion
        if self.fusion:
            self.fc1 = nn.Linear(1900*3, h_units)
        else:
            self.fc1 = nn.Linear(1900, h_units)
        self.fc2 = nn.Linear(h_units, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.fusion:
            x = x.view(-1, 1900*3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
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
        self.bn1d1 = nn.BatchNorm1d(16)
        self.bn1d2 = nn.BatchNorm1d(8)
        self.bn1d3 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = x.view(-1, 1, 3, 1900)
        x = self.conv1(x)
        x = x.view(-1, 16, 1900)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.bn1d1(x)
        x = self.conv2(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = self.bn1d2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = self.bn1d3(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class Bn_prelu(nn.Module):

    def __init__(self, num_features):
        super(Bn_prelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x

class DenseNN(nn.Module):
    '''Dense NN, using 3x3 filters, prelu and batch norm'''

    def __init__(self):
        super(DenseNN, self).__init__()
        # dense layers
        self.conv_init = nn.Conv2d(1, 16, 3, padding=(1, 1))
        self.bnp1 = Bn_prelu(16)
        self.conv1 = nn.Conv2d(16, 8, 3, padding=(1, 1))
        self.bnp2 = Bn_prelu(24)
        self.conv2 = nn.Conv2d(24, 8, 3, padding=(1, 1))
        self.bnp3 = Bn_prelu(32)
        self.conv3 = nn.Conv2d(32, 8, 3, padding=(1, 1))
        self.bnp4 = Bn_prelu(40)

        # compression layers
        # maybe add more layers to reduce length (1900) of vector?

        # pooling
        self.gapool = nn.AvgPool2d(kernel_size=(3, 1900))

        self.fc1 = nn.Linear(40, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x_0 = self.conv_init(x)
        x_1 = self.bnp1(x_0)
        x_1 = self.conv1(x_1)
        x_1_ = torch.cat((x_0, x_1), dim=1)
        x_2 = self.bnp2(x_1_)
        x_2 = self.conv2(x_2)
        x_2_ = torch.cat((x_0, x_1, x_2), dim=1)
        x_3 = self.bnp3(x_2_)
        x_3 = self.conv3(x_3)
        xfin = torch.cat((x_0, x_1, x_2, x_3), dim=1)

        xfin = self.bnp4(xfin)

        xfin = self.gapool(xfin)
        xfin = torch.flatten(xfin, start_dim=1)

        xout = self.fc1(xfin)

        return xout

class SimpleNN3(nn.Module):
    '''Simple 3 layer neural net with h_units amount of hidden units, flattens the input 3 layer vector
    '''
    def __init__(self, h_units1=64, h_units2=64, fusion=True):
        super(SimpleNN3, self).__init__()
        self.fusion = fusion
        if self.fusion:
            self.fc1 = nn.Linear(1900*3, h_units1)
        else:
            self.fc1 = nn.Linear(1900, h_units1)
        self.fc2 = nn.Linear(h_units1, h_units2)
        self.fc3 = nn.Linear(h_units2, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(h_units1)
        self.bn2 = nn.BatchNorm1d(h_units2)

    def forward(self, x):
        if self.fusion:
            x = x.view(-1, 1900*3)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



