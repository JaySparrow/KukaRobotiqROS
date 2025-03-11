import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x # self.softmax(x)
    
class CustomMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(CustomMLP, self).__init__()

        ## Architecture
        self.layers = nn.ModuleList()
        # batch normalization of input
        self.layers.append(nn.BatchNorm1d(input_dim))
        prev_dim = input_dim
        # input + hidden layers
        if hidden_dims is not None and len(hidden_dims) > 0:
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.5))
                prev_dim = hidden_dim
        # output layer
        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
    
    def forward(self, x):
        return torch.tensor([[1.0, 0.0]]).repeat(x.size(0), 1).to(x.device)
    
class ThresholdModel(nn.Module):
    def __init__(self, threshold):
        super(ThresholdModel, self).__init__()
        self.threshold = threshold
    
    def forward(self, x):
        pred = torch.zeros(x.size(0), 2).to(x.device)
        pred[x[:, -1]>= self.threshold, 0] = 1.0
        pred[x[:, -1]< self.threshold, 1] = 1.0
        return pred