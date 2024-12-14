import torch
import torch.nn as nn
from torch import FloatTensor
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        std = nn.functional.softplus(self.std(x))
        return mean, std
    
    @torch.no_grad()
    def take_np_action(self, state: np.array):
        state = FloatTensor(state)
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.sample().cpu().numpy()

    def take_action(self, states, gradient = False):
        if gradient:
            mean, std = self.forward(states)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()
        else:
            with torch.no_grad():
                mean, std = self.forward(states)
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
        return actions
    
def CreatePolicyNet(env, hidden_dim=32):
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy_net = PolicyNet(state_dim, act_dim, hidden_dim)
    return policy_net