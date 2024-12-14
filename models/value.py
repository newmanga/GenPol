import torch
import torch.nn as nn
from torch import FloatTensor
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np

class ValueNet(nn.Module):
    def __init__(self, 
                 state_dim, 
                 hidden_sizes = (50,50,50),
                 ActivationFn = nn.Tanh,
                 lr=0.005):
        super(ValueNet, self).__init__()
        self.state_dim = state_dim

        # Build the network
        layers = []
        input_size = self.state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(ActivationFn())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))  # Output layer
        self.network = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, states):
        return self.network(states)