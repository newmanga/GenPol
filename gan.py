import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from models.GAN_models import *

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
    epochs = 1
    batch_size = 32
    disc_lr = 1e-4
    pol_lr = 1e-3 #0.02
    disc_int_epochs = 1
    pol_int_epochs = 10
    hidden_dim = 32
    noise_scale = 0.02
    shuffle = True
    objective = "regular"

    losses = {"disc":[], "pol":[],"p_std":[]}

    dataloader = get_dataloader(batch_size, shuffle=shuffle)
    env_dim = get_env_dim(dataloader)
    discriminator = MLPDiscriminator(env_dim, 
                                    learning_rate=disc_lr, 
                                    hidden_sizes=(hidden_dim,hidden_dim), 
                                    objective=objective, 
                                    num_epochs_per_step=disc_int_epochs)
    policy = GaussianMLPPolicy(env_dim, 
                            learning_rate=pol_lr, 
                            hidden_sizes=(hidden_dim,hidden_dim), 
                            objective=objective,
                            noise_scale=noise_scale,
                            num_epochs_per_step=pol_int_epochs)
    train_gan(epochs, dataloader, discriminator, policy, losses)