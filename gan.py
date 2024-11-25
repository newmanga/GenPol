import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *

class TrajectoryDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, normalize=True):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states

        if normalize:
            # Compute normalization parameters
            self.state_mean = self.states.mean(dim=0)
            self.state_std = self.states.std(dim=0) + 1e-8  # Add epsilon to avoid division by zero
            self.action_mean = self.actions.mean(dim=0)
            self.action_std = self.actions.std(dim=0) + 1e-8
            self.reward_mean = self.rewards.mean()
            self.reward_std = self.rewards.std() + 1e-8

            # Normalize data
            self.states = (self.states - self.state_mean) / self.state_std
            self.actions = (self.actions - self.action_mean) / self.action_std
            self.rewards = (self.rewards - self.reward_mean) / self.reward_std
            self.next_states = (self.next_states - self.state_mean) / self.state_std
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "state_mean": self.state_mean,
            "state_std": self.state_std
        }

def get_dataset():
    states = torch.load(EXPERT_DATA_STATES_PATH.as_posix())
    actions = torch.load(EXPERT_DATA_ACTIONS_PATH.as_posix())
    rewards = torch.load(EXPERT_DATA_REWARDS_PATH.as_posix())
    next_states = torch.load(EXPERT_DATA_NEXT_STATES_PATH.as_posix())

    dataset = TrajectoryDataset(states, actions, rewards, next_states)
    return dataset

def get_dataloader(batch_size, shuffle=True):
    dataset = get_dataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_env_dim(dataloader):
    state_dim = 0
    action_dim = 0
    for batch in dataloader:
        state_dim = batch["state"].shape[1]
        action_dim = batch["action"].shape[1]
        break
    return {"states": state_dim, "actions": action_dim}

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
    
    def get_std(self, states):
        with torch.no_grad():
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
                 num_epochs_per_step=1):
        super(MLPDiscriminator, self).__init__()
        
        self.ent_reg_weight = ent_reg_weight #TODO update name for consistency
        self.gradient_penalty_weight = gradient_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight
        self.objective = objective
        self.num_epochs_per_step = num_epochs_per_step

        # Build network
        layers = []
        input_size = env_dim["states"] + env_dim["actions"]
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(activation())
            input_size = size
        layers.append(nn.Linear(input_size, 1))  # Output layer
        self.network = nn.Sequential(*layers)

        # Loss and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits

    def forward(self, states_actions):
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

    def get_prediction(self, batch):
        with torch.no_grad():
            logits = self.forward(batch)
            return logits.sigmoid().numpy()
        
def step(states, actions, discriminator, policy):
    # --- Train Discriminator ---
    loss_discriminator = discriminator.train_step(states, actions, policy)

    # --- Train Generator ---
    loss_generator, std_generator = policy.train_step(states, discriminator)

    return loss_discriminator, loss_generator, std_generator

def train_policy(epochs, dataloader, discriminator, policy, losses):
    total_batches = len(dataloader) * epochs
    with tqdm(total=total_batches) as pbar:
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                p_loss, std_generator = policy.train_step(batch["state"], discriminator)
                if batch_idx % 1000 == 0 or batch_idx == len(dataloader) - 1:
                    losses["pol"].append(p_loss)
                    losses["std"].append(std_generator)
                    pbar.set_description(
                        f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                        f"P_Loss [{p_loss:.4f}]"
                    )
                pbar.update(1)

def train_disc(epochs, dataloader, discriminator, policy, losses):
    total_batches = len(dataloader) * epochs
    with tqdm(total=total_batches) as pbar:
        for s in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                loss = discriminator.train_step(batch["state"], batch["action"], policy)
                if batch_idx % 1000 == 0 or batch_idx == len(dataloader) - 1:
                    losses["disc"].append(loss)
                    pbar.set_description(
                        f"Step [{s+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], loss [{loss:.4f}]"
                    )
                pbar.update(1)

def train_gan(epochs, dataloader, discriminator, policy, losses):
    total_batches = len(dataloader) * epochs
    with tqdm(total=total_batches) as pbar:
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                d_loss , p_loss, p_std = step(batch["state"], batch["action"], discriminator, policy)

                if batch_idx % 1000 == 0 or batch_idx == len(dataloader) - 1:
                    losses["disc"].append(d_loss)
                    losses["pol"].append(p_loss)
                    losses["p_std"].append(p_std.mean(dim=0))
                    pbar.set_description(
                        f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                        f"D_Loss [{d_loss:.4f}], P_Loss [{p_loss:.4f}]"
                    )
                pbar.update(1)

def run_gan():
    epochs = 100
    batch_size = 64
    disc_lr = 1e-7
    pol_lr = 0.01
    hidden_dim = 32

    dataloader = get_dataloader(batch_size)
    env_dim = get_env_dim(dataloader)
    discriminator = MLPDiscriminator(env_dim, learning_rate=disc_lr)
    policy = GaussianMLPPolicy(env_dim, learning_rate=pol_lr, hidden_sizes=(hidden_dim,hidden_dim))
    return train_gan(epochs, dataloader, discriminator, policy)