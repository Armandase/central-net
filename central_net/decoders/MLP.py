import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=0, dropout_rate=0):
        super(MLP, self).__init__()
        if hidden_dim == 0:
            hidden_dim = input_dim // 4
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# mlp to handle tensor like this torch.Size([1, 2048, 7, 7])


class MLP_2d(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=0, dropout_rate=0):
        super(MLP_2d, self).__init__()
        if hidden_dim == 0:
            hidden_dim = input_dim // 4
        width = 7
        input_dim = input_dim * width * width
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        print("input mlp:", x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        print("output mlp:", x.shape)
        return x
