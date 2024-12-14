import torch
import torch.nn as nn
from torch import FloatTensor
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np

from models.base import PolicyBase

class GaussianPolicyNet(nn.Module, PolicyBase):
    def __init__(self, 
                 env_dim, 
                 hidden_sizes=(400, 300),
                 ActivationFn=nn.Tanh,
                 learning_rate=1e-4,
                 state_independent_log_std=True,
                 min_std=1e-6, 
                 std_init = 0.606):
        super().__init__()

        self.state_dim = env_dim["states"]
        self.action_dim = env_dim["actions"]
        self.hidden_sizes = hidden_sizes
        self.state_independent_log_std = state_independent_log_std
        self.min_std_param = np.log(min_std)
        self.log_std_init = np.log(std_init)

        # Build the network
        layers = []
        input_size = self.state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(ActivationFn())
            input_size = hidden_size
        self.feature_extractor = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mu_layer = nn.Linear(hidden_sizes[-1], self.action_dim)

        if self.state_independent_log_std:
            self.log_std = nn.Parameter(torch.full((self.action_dim,), self.log_std_init))
        else:
            self.log_std_layer = nn.Linear(hidden_sizes[-1], self.action_dim)
        
        # Loss and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, state):
        features = self.feature_extractor(state)
        mu = self.mu_layer(features)

        if self.state_independent_log_std:
            log_std = self.log_std
        else:
            log_std = self.log_std_layer(features)
            log_std = torch.clamp(log_std, min=self.min_std_param, max=0.0)
        return mu, log_std
    
    def get_dist(self, state: np.array):
        self.eval()
        state = FloatTensor(state)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)
        return MultivariateNormal(mean, cov_mtx)
    
    def get_dist_t(self, state: torch.tensor):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)
        return MultivariateNormal(mean, cov_mtx)
    
    @torch.no_grad()
    def take_np_action(self, state: np.array):
        dist = self.get_dist(state)
        return dist.sample().cpu().numpy()
    
def init_weights_kaiming(m):
    """
    Custom weight initialization using Kaiming initialization.
    Suitable for ReLU or LeakyReLU activations.
    """
    if isinstance(m, nn.Linear):
        # Apply Kaiming initialization for weights
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        # Initialize biases to zero
        if m.bias is not None:
            nn.init.zeros_(m.bias)