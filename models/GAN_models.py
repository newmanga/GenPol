import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *

class GaussianMLPPolicy(nn.Module):
    def __init__(self, 
                 env_dim, 
                 hidden_sizes=(400, 300),
                 noise_scale=0.02, 
                 min_std=1e-6, 
                 learning_rate=1e-4,
                 objective="regular",
                 num_epochs_per_step=3,
                 state_independent_log_std=True,
                 std_init = 0.606):
        super().__init__()

        self.objective = objective
        self.num_epochs_per_step = num_epochs_per_step
        self.hidden_sizes = hidden_sizes
        self.noise_scale = noise_scale
        self.state_independent_log_std = state_independent_log_std
        self.min_std_param = np.log(min_std)
        self.log_std_init = np.log(std_init)

        # Build the network
        layers = []
        input_size = env_dim["states"]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU())
            input_size = hidden_size
        self.feature_extractor = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mu_layer = nn.Linear(hidden_sizes[-1], env_dim["actions"])

        if self.state_independent_log_std:
            self.log_std = nn.Parameter(torch.full((env_dim["actions"],), self.log_std_init))
        else:
            self.log_std_layer = nn.Linear(hidden_sizes[-1], env_dim["actions"])
        
        # mu_layer initialization
        self.mu_layer.weight.data.mul_(0.1)
        self.mu_layer.bias.data.mul_(0.0)

        # Loss and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, state):
        features = self.feature_extractor(state)
        mu = self.mu_layer(features)
        # mu = torch.tanh(self.mu_layer(features))
        if self.state_independent_log_std:
            log_std = self.log_std.expand_as(mu)
        else:
            log_std = self.log_std_layer(features)
            log_std = torch.clamp(log_std, min=self.min_std_param, max=0.0)
        return mu, log_std
    
    def noisy_states(self, states):
        return states + torch.randn_like(states) * self.noise_scale
    
    def take_action(self, states, gradient = False):
        if gradient:
            mean, log_std = self.forward(states)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
        else:
            with torch.no_grad():
                mean, log_std = self.forward(states)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        return action
    
    @torch.no_grad()
    def get_std(self, states):
        _, log_std = self.forward(states)
        std = torch.exp(log_std)
        return std
        
    def train_step(self, states, discriminator):
        batch_size = states.shape[0]
        for i in range(self.num_epochs_per_step):

            self.optimizer.zero_grad()

            noisy_states = self.noisy_states(states)
            real_labels = torch.ones(batch_size, 1)

            fake_actions = self.take_action(noisy_states, gradient=True)
            fake_state_action = torch.cat([noisy_states, fake_actions], dim=-1)

            if self.objective == "wgan":
                loss = -discriminator(fake_state_action).mean()
            else:
                loss = self.loss_fn(discriminator(fake_state_action), real_labels)

            retain = i < (self.num_epochs_per_step - 1)
            loss.backward(retain_graph=retain)
            self.optimizer.step()
        
        return loss.item(), self.get_std(noisy_states)
    
class MLPDiscriminator(nn.Module):
    def __init__(self,
                 env_dim,
                 hidden_sizes=(400, 300),
                 activation=nn.Tanh,
                 learning_rate=1e-4,
                 ent_reg_weight=0.0,
                 gradient_penalty_weight=0.0,
                 l2_penalty_weight=0.001,
                 objective="regular",
                 num_epochs_per_step=1,
                 diffusion=True,
                 t_max=100,
                 t_rate=1000):
        super(MLPDiscriminator, self).__init__()
        
        self.ent_reg_weight = ent_reg_weight #TODO update name for consistency
        self.gradient_penalty_weight = gradient_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight
        self.objective = objective
        self.num_epochs_per_step = num_epochs_per_step
        self.diffusion = diffusion

        # Build network
        layers = []
        input_size = env_dim["states"] + env_dim["actions"] + (1 if self.diffusion else 0)
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(activation())
            input_size = size
        layers.append(nn.Linear(input_size, 1))  # Output layer
        self.network = nn.Sequential(*layers)

        # Loss and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits

        ## Diffusion
        self.p = 0.0
        self.t_min = 0
        self.t_max = t_max
        self.t = 0.0
        self.noise_std = 1.0
        self.t_rate = t_rate
        self.t_epl = np.zeros(64, dtype=np.int32)
        self.set_difussion()

    def forward(self, states_actions):

        if self.diffusion:

            t = torch.from_numpy(np.random.choice(self.t_epl, size=states_actions.shape[0], replace=True)).unsqueeze(-1)
            noise = torch.randn_like(states_actions) * self.noise_std

            alphas_t_sqrt = self.alphas_bar_sqrt[t]
            one_minus_alphas_bar_t_sqrt = self.one_minus_alphas_bar_sqrt[t]

            states_actions = alphas_t_sqrt * states_actions + one_minus_alphas_bar_t_sqrt * noise

            states_actions = torch.cat((states_actions, t), dim=1)

        return self.network(states_actions)

    def compute_loss(self, logits, targets):
        if self.objective == "wgan":
            disc_fake = logits[:-targets.sum().int()]
            disc_real = logits[-targets.sum().int():]
            loss = disc_fake.mean() - disc_real.mean()
        else:
            entropy = -(logits.sigmoid() * torch.log(logits.sigmoid() + 1e-8)).mean()
            cross_entropy = self.loss_fn(logits, targets)
            loss = cross_entropy - self.ent_reg_weight * entropy

        # L2 penalty
        if self.l2_penalty_weight > 0.0:
            l2_loss = sum(torch.norm(param) for param in self.parameters()) * self.l2_penalty_weight
            loss += l2_loss

        return loss

    def get_states_actions(self, states, actions, policy):
        noise_input = policy.noisy_states(states)
        fake_actions = policy.take_action(noise_input).detach()
        fake_state_action = torch.cat([noise_input, fake_actions], dim=-1)

        real_state_action = torch.cat([states, actions], dim=-1)

        state_action = torch.cat([fake_state_action, real_state_action], dim=0)
        return state_action

    def get_labels(self, states):
        batch_size = states.shape[0]
        labels = torch.zeros((batch_size*2,1))
        labels[batch_size:] = 1.0
        return labels

    def train_step(self, states, actions, policy, clip_weights=False):
        for _ in range(self.num_epochs_per_step):
            state_action = self.get_states_actions(states, actions, policy)
            labels = self.get_labels(states)
            self.optimizer.zero_grad()
            logits = self.forward(state_action)
            loss = self.compute_loss(logits, labels)
            loss.backward()
            self.optimizer.step()

            # Clip weights for WGAN
            if self.objective == "wgan":
                for param in self.parameters():
                    param.data.clamp_(-0.01, 0.01)

        return loss.item()

    @torch.no_grad()
    def get_prediction(self, states, actions):
        real_state_action = torch.cat([states, actions], dim=-1)
        logits = self.forward(real_state_action)
        return logits.sigmoid().numpy()
    
    def update_times(self, real_pred, target = 0.6):
        overfitting_metric = np.sign(real_pred - target)
        adjust = overfitting_metric / self.t_rate
        self.p = (self.p + adjust).clip(min=0., max=1.)
        self.t = self.t_min + int((self.t_max - self.t_min) * self.p)
        self.update_t_epl()
        self.set_difussion()

    def update_t_epl(self):
        ts_dist = 'priority'
        self.t_epl = np.zeros(64, dtype=np.int32)
        if (self.t <= 1):
            return
        diffusion_ind = 32
        t_diffusion = np.zeros((diffusion_ind,)).astype(np.int32)
        if ts_dist == 'priority':
            prob_t = np.arange(self.t+1) / np.arange(self.t+1).sum()
            t_diffusion = np.random.choice(np.arange(0, self.t + 1), size=diffusion_ind, p=prob_t)
        elif ts_dist == 'uniform':
            t_diffusion = np.random.choice(np.arange(0, self.t + 1), size=diffusion_ind)
        self.t_epl[:diffusion_ind] = t_diffusion

    def set_difussion(self):
        beta_start = 01e-4
        beta_end = 2e-2
        betas = torch.linspace(beta_start, beta_end, int(self.t))

        alphas = 1.0 - betas
        alphas_cumprod = torch.cat([torch.tensor([1.]), alphas.cumprod(dim=0)])
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)