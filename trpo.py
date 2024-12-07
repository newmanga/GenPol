import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import namedtuple
from tqdm import tqdm
from collections import deque
from gym_env.environments import *

from utils import to_tensor, normalize
import time

# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")
torch.manual_seed(0)

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states'])

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

    def take_action(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, lr=0.005):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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

def CreatePolicyNet(env, hidden_dim=32):
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy_net = PolicyNet(state_dim, act_dim, hidden_dim)
    # policy_net.apply(init_weights_kaiming)
    return policy_net

def CreateValueNet(env, hidden_dim=32, value_learning_rate=0.005):
    state_dim = env.observation_space.shape[0]
    value_net = ValueNetwork(state_dim, hidden_dim, value_learning_rate)
    # value_net.apply(init_weights_kaiming)
    return value_net

def collect_trajectories(params: dict, env: Env, policy_net: PolicyNet, log = False):

    def to_tensor(A):
        return torch.tensor(A, dtype=torch.float32).unsqueeze(0)

    trajectories = []
    trajectories_total_reward = []

    for _ in range(params["num_episodes"]):
        state = to_tensor(env.reset()[0])
        done = False
        trunc = False
        trajectory = []
        
        num_steps = 0
        while not done and not trunc and num_steps < params["max_num_steps"]:
            action = policy_net.take_action(state)
            next_state, reward, done, trunc = env.take_step(state, action)
            next_state = to_tensor(next_state)

            trajectory.append((state, action, reward, next_state))
            state = next_state
            num_steps += 1

        states, actions, rewards, next_states = zip(*trajectory)

        states = torch.stack(states).squeeze(1)
        next_states = torch.stack(next_states).squeeze(1)
        actions = torch.stack(actions).squeeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)

        trajectories.append(Rollout(states, actions, rewards, next_states))

        trajectories_total_reward.append(rewards.sum().item())
        
    mean_total_reward = np.mean(trajectories_total_reward)
    return trajectories, mean_total_reward

class TRPO():
    def __init__(self, env: gym.Env, policy_net: PolicyNet, value_net: ValueNetwork, params: dict):
        self.env = env
        self.params = params
        self.policy_net = policy_net
        self.value_net = value_net

    def compute_surrogate_loss(self, new_probabilities, old_probabilities, advantages, entr, entr_coef=0.01):
        ratios = torch.exp(new_probabilities - old_probabilities)
        return (ratios * advantages).mean() - (entr_coef * entr)



    def compute_td_error(self, trajectories, gamma):
        """Calculate Temporal-Difference Error"""

        rewards = torch.cat([r.rewards for r in trajectories], dim=0).float()
        states = torch.cat([r.states for r in trajectories], dim=0).float()
        next_states = torch.cat([r.next_states for r in trajectories], dim=0).float()

        td_error = rewards + gamma * self.value_net(next_states) - self.value_net(states)
        return td_error

    def compute_advantages(self, trajectories, gamma=0.9, lmbda=0.9):
        """Generalized Advantage Estimator"""

        size = sum(r.rewards.size(0) for r in trajectories)

        td_error = self.compute_td_error(trajectories, gamma)

        advantages = torch.zeros(size, dtype=torch.float)
        advantages[size - 1] = td_error[size - 1]

        for t in range(size - 2, -1, -1):
            advantages[t] = td_error[t] + (gamma * lmbda * advantages[t + 1])

        return advantages.unsqueeze(-1)

    def update_policy(self, trajectories, advantages):

        states = torch.cat([r.states for r in trajectories], dim=0).float()
        actions = torch.cat([r.actions for r in trajectories], dim=0).float()

        mean, std = self.policy_net(states)
        dist_old = torch.distributions.Normal(mean.detach(), std.detach())
        prob_old = dist_old.log_prob(actions)

        def get_loss():
            mean_new, std_new = self.policy_net(states)
            dist_new = torch.distributions.Normal(mean_new, std_new)
            prob_new = dist_new.log_prob(actions)
            entr = dist_new.entropy().mean()

            return {"kl":torch.distributions.kl.kl_divergence(dist_old, dist_new).mean(),
                    "surrogate_loss":self.compute_surrogate_loss(prob_new, prob_old.detach(), advantages, entr, self.params["entr_coeff"])}
        
        def flat_grad(y, x, retain_graph=False, create_graph=False):
            if create_graph:
                retain_graph = True

            g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
            g = torch.cat([t.view(-1) for t in g])
            return g
        
        #Policy Gradient
        g = flat_grad(get_loss()["surrogate_loss"], self.policy_net.parameters(), retain_graph=True)
        
        def HVP(v):
            """Hessian Vector Product"""
            flat_grads = flat_grad(get_loss()["kl"], self.policy_net.parameters(), create_graph=True)
            inner = flat_grads @ v
            return flat_grad(inner, self.policy_net.parameters(), retain_graph=True) + v * self.params["damping_coeff"]

        search_dir = self.conjugate_gradient(HVP, g)
        quadratic_term = (search_dir @ HVP(search_dir)).sum()
        beta = torch.sqrt(2 * self.params["max_d_kl"] / (quadratic_term + 1e-6)) #max length
        max_step = beta * search_dir

        self.line_search(get_loss, max_step, self.params["max_d_kl"])

    def line_search(self, get_loss, max_step, max_d_kl):
        with torch.no_grad():
            L_old = get_loss()["surrogate_loss"]
            alpha = 0.9
            for i in range(10):
                step = max_step * (alpha ** i)
                self.apply_update(step)

                l = get_loss()
                L_new = l["surrogate_loss"]
                kl_new = l["kl"]

                if (L_new - L_old) > 0 and kl_new <= max_d_kl:
                    break

                self.apply_update(-step)

    def conjugate_gradient(self, A, b, delta=1e-6, max_iterations=10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = r @ r

        for i in range(max_iterations):
            AVP = A(p)
            alpha = rdotr / (p @ AVP)

            x += alpha * p
            r -= alpha * AVP
            new_rdotr = r @ r

            if new_rdotr < delta:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def update_value_net(self, trajectories, gamma=0.99):
        states = torch.cat([r.states for r in trajectories], dim=0).float()
        td_target = self.compute_td_error(trajectories, gamma)
        loss = nn.functional.mse_loss(self.value_net(states), td_target.detach()).mean()

        # Perform gradient descent
        self.value_net.optimizer.zero_grad()
        loss.backward()
        self.value_net.optimizer.step()

    def train_trpo(self):

        last_10_returns = deque(maxlen=10)

        with tqdm(total=self.params["epochs"]) as pbar:
            for _ in range(self.params["epochs"]):
                trajectories, mtr = collect_trajectories(self.params, self.env, self.policy_net)
                advantages = self.compute_advantages(trajectories)

                self.update_value_net(trajectories)

                self.update_policy(trajectories, advantages)

                # Update the deque with the latest reward
                last_10_returns.append(mtr)

                # Calculate the running average
                running_avg = sum(last_10_returns) / len(last_10_returns)

                pbar.set_postfix({
                    'return': f'{mtr:.3f}',
                    'avg_return': f'{running_avg:.3f}'
                })
                pbar.update(1)

    def apply_update(self, grad_flattened):
        n = 0
        for p in self.policy_net.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

def try_policy(policy, render=True, normalize_state=False, mean=0, std=1, max_num_steps=200, time_delay=0.1):
    env = create_halfcheetah_env(render, 1)
    state = to_tensor(env.reset()[0])
    if normalize_state:
        state = normalize(state, mean, std)
    done = False
    trunc = False
    sum_rewards = 0
    trajectory = []

    num_steps = 0
    while not done and not trunc and num_steps < max_num_steps:

        action = policy.take_action(state)
        next_state, reward, done, trunc = env.take_step(state, action)
        next_state = to_tensor(next_state)
        if normalize_state:
            next_state = normalize(next_state, mean, std)
        sum_rewards += reward

        trajectory.append((state, action, reward, next_state))
        state = next_state

        num_steps += 1

        time.sleep(time_delay)  # Simulate a delay
    return sum_rewards/num_steps

def get_total_rewards(policy, states):
    actions = policy.take_action(states)



# Example usage
if __name__ == '__main__':

    hyperparams = {"epochs":1000, 
                   "env_seed":1, 
                   "num_episodes":10, 
                   "max_num_steps":500, 
                   "max_d_kl" : 0.01, 
                   "damping_coeff" : 0.1, 
                   "entr_coeff" : 0.03}

    render = False
    fwd_reward_w = 1
    env = create_halfcheetah_env(render, fwd_reward_w)

    value_learning_rate = 0.005
    hidden_dim = 64

    #Create NNs
    policy_net = CreatePolicyNet(env, hidden_dim)
    value_net = CreateValueNet(env, hidden_dim, value_learning_rate)

    trpo = TRPO(env, policy_net, value_net, hyperparams) 
    trpo.train_trpo()

    #Render final policy
    env = create_halfcheetah_env(render=True, forward_reward_weight=fwd_reward_w)
    collect_trajectories(env, policy_net, hyperparams, log=False)
