import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.diffusion import Diffusion

class DiscriminatorNet(nn.Module):
    def __init__(self,
                 env_dim,
                 config,
                 diffusion=None):
        super(DiscriminatorNet, self).__init__()

        # Configuration parameters
        self.hidden_sizes               = config.get("hidden_sizes", (400, 300))
        activation                      = config.get("activation", nn.Tanh)
        self.learning_rate              = config.get("disc_lr", 1e-3)
        self.ent_reg_weight             = config.get("ent_reg_weight", 0.0)
        self.gradient_penalty_weight    = config.get("gradient_penalty_weight", 0.0)
        self.l2_penalty_weight          = config.get("l2_penalty_weight", 0.001)
        self.objective                  = config.get("objective", "regular")

        self.diffusion : Diffusion = diffusion

        # Build network
        layers = []
        input_size = env_dim["states"] + env_dim["actions"] + (1 if self.diffusion else 0)
        for size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(activation())
            input_size = size
        layers.append(nn.Linear(input_size, 1))  # Output layer
        self.network = nn.Sequential(*layers)

        # Loss and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, states_actions):
        if self.diffusion:
            states_actions = self.diffusion.forward_process(states_actions)

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

        if self.l2_penalty_weight > 0.0:
            l2_loss = sum(torch.norm(param) for param in self.parameters()) * self.l2_penalty_weight
            loss += l2_loss

        return loss

    def get_logits(self, states, actions):
        states_actions = torch.cat([states, actions], dim=-1)
        return self.forward(states_actions)

    def train_step(self, exp_sta, exp_act, gen_sta, gen_act):
        self.train()
        exp_scores = self.get_logits(exp_sta, exp_act)
        gen_scores = self.get_logits(gen_sta, gen_act)
        self.optimizer.zero_grad()
        prob_real_being_real = self.loss_fn(exp_scores, torch.ones_like(exp_scores)) 
        prob_fake_being_fake = self.loss_fn(gen_scores, torch.zeros_like(gen_scores))
        loss = prob_real_being_real + prob_fake_being_fake
        loss.backward()
        self.optimizer.step()
        return prob_real_being_real.item(), prob_fake_being_fake.item()

    @torch.no_grad()
    def get_prediction(self, states, actions):
        real_state_action = torch.cat([states, actions], dim=-1)
        logits = self.forward(real_state_action)
        return logits.sigmoid().numpy()
    
    def update_times(self, real_pred):
        if self.diffusion:
            self.diffusion.update(real_pred)