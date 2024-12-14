import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Diffusion:
    def __init__(self, config=dict()):
        
        # Configuration parameters
        self.t_min = config.get("t_min", 0)
        self.t_max = config.get("t_max", 100)
        self.t_rate = config.get("t_rate", 1000)
        self.ts_dist = config.get("ts_dist", 'priority')
        self.beta_start = config.get("beta_start", 1e-4)
        self.beta_end = config.get("beta_end", 2e-2)
        self.target = config.get("target", 0.6)
        
        self.p = 0.0
        self.t = 0.0
        self.noise_std = 1.0
        self.t_epl = np.zeros(64, dtype=np.int32)
        self.set_diffusion()

    def update(self, real_pred):
        overfitting_metric = np.sign(real_pred - self.target)
        adjust = overfitting_metric / self.t_rate
        self.p = (self.p + adjust).clip(min=0., max=1.)
        self.t = self.t_min + int((self.t_max - self.t_min) * self.p)
        self.update_t_epl()
        self.set_diffusion()

    def update_t_epl(self):
        self.t_epl = np.zeros(64, dtype=np.int32)
        if self.t <= 1:
            return
        diffusion_ind = 32
        t_diffusion = np.zeros((diffusion_ind,)).astype(np.int32)
        if self.ts_dist == 'priority':
            prob_t = np.arange(self.t+1) / np.arange(self.t+1).sum()
            t_diffusion = np.random.choice(np.arange(0, self.t + 1), size=diffusion_ind, p=prob_t)
        elif self.ts_dist == 'uniform':
            t_diffusion = np.random.choice(np.arange(0, self.t + 1), size=diffusion_ind)
        self.t_epl[:diffusion_ind] = t_diffusion

    def set_diffusion(self):
        betas = torch.linspace(self.beta_start, self.beta_end, int(self.t))
        alphas = 1.0 - betas
        alphas_cumprod = torch.cat([torch.tensor([1.]), alphas.cumprod(dim=0)])
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

    def forward_process(self, states_actions):
        t = torch.from_numpy(np.random.choice(self.t_epl, size=states_actions.shape[:-1], replace=True)).unsqueeze(-1)
        noise = torch.randn_like(states_actions) * self.noise_std

        alphas_t_sqrt = self.alphas_bar_sqrt[t]
        one_minus_alphas_bar_t_sqrt = self.one_minus_alphas_bar_sqrt[t]

        states_actions = alphas_t_sqrt * states_actions + one_minus_alphas_bar_t_sqrt * noise
        states_actions = torch.cat((states_actions, t), dim=-1)

        return states_actions