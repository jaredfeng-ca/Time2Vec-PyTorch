from periodic_activations import t2v
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, activation, in_features, hidden_dim, n_classes):
        super(Model, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(in_features, hidden_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(in_features, hidden_dim)
        
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, n_classes), nn.Softmax())
    
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x