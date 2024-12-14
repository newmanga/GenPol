import numpy as np
import torch
from torch import FloatTensor
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import namedtuple
from gym_env.environments import *
import time

from models.policy import *

class TRPO():
    def __init__(self, 
                 action_dim, 
                 policy_net, 
                 value_net,
                 config):
        
        self.action_dim = action_dim
        self.policy_net = policy_net
        self.value_net = value_net
        self.lambda_ = config["entropy_w"]
        self.gae_gamma = config["gae_gamma"]
        self.gae_lmbda = config["gae_lambda"]
        self.normalize_advantage = config["normalize_advantage"]
        self.eps = config["epsilon"]
        self.max_d_kl = config["max_kl"]
        self.damping_coeff = config["damping_coeff"]

    def get_discounts(self, num_state_actions):
        gammas = np.ones(num_state_actions)
        lambdas = np.ones(num_state_actions)
        for i in range(1,num_state_actions):
            gammas[i] = self.gae_gamma * gammas[i-1]
            lambdas[i] = self.gae_lmbda * lambdas[i-1]
        return gammas, lambdas

    def get_returns(self, log_prob, gammas): ##TODO remove gammas operations
        disc_costs = gammas * log_prob

        disc_returns = torch.flip(torch.cumsum(torch.flip(disc_costs, dims=[-1]), dim=-1), dims=[-1])
        
        return disc_returns / gammas

    def get_td_error(self, states, rewards):
        self.value_net.eval()
        curr_vals = self.value_net(states).detach()
        if curr_vals.dim() == 2:
            zero_padding = torch.zeros(1, curr_vals.size(1))
            next_vals = torch.cat(
                (curr_vals[1:], zero_padding)
            ).detach()
        elif curr_vals.dim() == 3: 
            batch_size, seq_len, num_features = curr_vals.shape
            # Handle next_vals per batch in a vectorized way
            zero_padding = torch.zeros((batch_size, 1, num_features))
            next_vals = torch.cat(
                (curr_vals[:, 1:, :], zero_padding), dim=1
            ).detach()
        else:
            raise ValueError(f"Unexpected curr_vals shape: {curr_vals.shape}")
        
        return rewards + self.gae_gamma * next_vals - curr_vals
    
    def get_advantages(self, deltas, gammas, lambdas, num_state_actions):

        batch_size, seq_len, _ = deltas.shape
        discount_factors = gammas * lambdas
        discount_factors = discount_factors.unsqueeze(-1)

        if deltas.dim() == 2:  # Shape (200, 1)
            advs = torch.FloatTensor([
                (discount_factors[:num_state_actions - j] * deltas[j:]).sum()
                for j in range(num_state_actions)
            ])#.unsqueeze(-1)
        elif deltas.dim() == 3:
            batch_size, seq_len, _ = deltas.shape
            advs_list = []
            for b in range(batch_size):  # Loop over batch
                batch_deltas = deltas[b]
                batch_advs = torch.FloatTensor([
                    (discount_factors[:num_state_actions - j] * batch_deltas[j:]).sum()
                    for j in range(num_state_actions)
                ])#.unsqueeze(-1)
                advs_list.append(batch_advs)
            advs = torch.stack(advs_list)  

        if self.normalize_advantage:
            if advs.dim() == 2:  
                advs = (advs - advs.mean()) / advs.std()
            elif advs.dim() == 3:  
                mean = advs.mean(dim=1, keepdim=True)
                std = advs.std(dim=1, keepdim=True)
                advs = (advs - mean) / std

        return advs

    def compute_surrogate_loss(self, new_probabilities, old_probabilities, advantages):
        ratios = torch.exp(new_probabilities - old_probabilities)
        return (ratios * advantages).mean()
    
    def update_value_net(self, states, rets):
        self.value_net.train()
        old_v = self.value_net(states).detach()

        def flat_grad(y, x, retain_graph=False, create_graph=False):
            if create_graph:
                retain_graph = True

            g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
            g = torch.cat([t.view(-1) for t in g])
            return g

        def constraint():
            return ((old_v - self.value_net(states)) ** 2).mean()

        grad_diff = flat_grad(constraint(), self.value_net.parameters(), create_graph=True)

        def HVP(v):
            hessian = flat_grad(torch.dot(grad_diff, v), self.value_net.parameters(), create_graph=True).detach()
            return hessian
        
        def apply_update(grad_flattened):
            n = 0
            for p in self.value_net.parameters():
                numel = p.numel()
                g = grad_flattened[n:n + numel].view(p.shape)
                p.data += g
                n += numel

        g = flat_grad(
            ((-1) * (self.value_net(states).squeeze() - rets) ** 2).mean(), self.value_net.parameters(), create_graph=True).detach()
        s = self.conjugate_gradient(HVP, g).detach()

        Hs = HVP(s).detach()
        alpha = torch.sqrt(2 * self.eps / torch.dot(s, Hs))

        apply_update(alpha * s)

    def kl_divergence(self, dist_old, states):
        distb = self.policy_net.get_dist(states)

        old_mean = dist_old.mean.detach()
        old_cov = dist_old.covariance_matrix.sum(-1).detach()
        mean = distb.mean
        cov = distb.covariance_matrix.sum(-1)

        return (0.5) * (
                (old_cov / cov).sum(-1)
                + (((old_mean - mean) ** 2) / cov).sum(-1)
                - self.action_dim
                + torch.log(cov).sum(-1)
                - torch.log(old_cov).sum(-1)
            ).mean()
    
    def update_policy_net(self, states, actions, advantages, gammas):
        self.policy_net.train()
        dist_old = self.policy_net.get_dist(states)
        prob_old = dist_old.log_prob(actions)

        def get_loss():
            dist_new = self.policy_net.get_dist(states)
            prob_new = dist_new.log_prob(actions)

            return {"kl":self.kl_divergence(dist_old, states),
                    "surrogate_loss":self.compute_surrogate_loss(prob_new, prob_old.detach(), advantages)}
        
        def flat_grad(y, x, retain_graph=False, create_graph=False):
            if create_graph:
                retain_graph = True

            g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
            g = torch.cat([t.view(-1) for t in g])
            return g
        
        #Policy Gradient
        g = flat_grad(get_loss()["surrogate_loss"], self.policy_net.parameters(), retain_graph=True).detach()

        def HVP(v):
            """Hessian Vector Product"""
            flat_grads_kl = flat_grad(get_loss()["kl"], self.policy_net.parameters(), create_graph=True)
            inner = torch.dot(flat_grads_kl, v)
            return flat_grad(inner, self.policy_net.parameters(), retain_graph=True).detach() + v * self.damping_coeff

        search_dir = self.conjugate_gradient(HVP, g)

        quadratic_term = (search_dir @ HVP(search_dir)).sum()
        beta = torch.sqrt(2 * self.max_d_kl / (quadratic_term + 1e-6)) #max length
        max_step = beta * search_dir

        self.line_search(get_loss, max_step, self.max_d_kl)

        ### --- Causal Entropy Update --- ###
        # 1. Compute causal entropy
        disc_causal_entropy = ((-1) * gammas * self.policy_net.get_dist(states).log_prob(actions)).mean()
        
        # 2. Compute gradient of causal entropy
        grad_disc_causal_entropy = flat_grad(disc_causal_entropy, self.policy_net.parameters())
        
        # 3. Apply the causal entropy gradient update
        self.apply_update(self.lambda_ * grad_disc_causal_entropy)

    def line_search(self, get_loss, max_step, max_d_kl):
        with torch.no_grad():
            updated = False
            L_old = get_loss()["surrogate_loss"]
            alpha = 0.9
            for i in range(10):
                step = max_step * (alpha ** i)
                self.apply_update(step)

                l = get_loss()
                L_new = l["surrogate_loss"]
                kl_new = l["kl"]

                if (L_new - L_old) > 0 and kl_new <= max_d_kl:
                    updated = True
                    break

                self.apply_update(-step)
            if not updated:
                print("not updated")

    def get_flat_params(self, net):
        return torch.cat([param.view(-1) for param in net.parameters()])

    def conjugate_gradient(self, A, b, delta=1e-10, max_iterations=10):
        x = torch.zeros_like(b)
        r = b.clone() - A(x)
        p = r
        rdotr = r @ r

        for i in range(max_iterations):
            AVP = A(p)
            alpha = rdotr / (p @ AVP)

            x += alpha * p
            r -= alpha * AVP
            new_rdotr = r @ r

            if torch.sqrt(new_rdotr) < delta:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def apply_update(self, grad_flattened):
        n = 0
        for p in self.policy_net.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def train_model(self, env, num_episodes):
        for episode in range(num_episodes):
            # Collect rollouts from the environment
            states_b, actions_b, rewards_b = env.get_rollout_tensors(self.policy_net,
                                                                        self.batch_size, 
                                                                        self.num_steps_per_iter)

            # Calculate the discount factors
            gammas, lambdas = map(lambda x: FloatTensor(x), self.get_discounts(len(states_b[0])))

            returns = self.get_returns(rewards_b, gammas)

            deltas = self.get_td_error(states_b, rewards_b)

            advantages = self.get_advantages(deltas, gammas, lambdas, self.num_steps_per_iter)
            
            # Update value network
            self.update_value_net(states_b, returns)

            # Update policy network
            self.update_policy_net(states_b, actions_b, advantages, gammas)

            print(f"Episode {episode + 1}/{num_episodes}, Return: {rewards_b.sum(dim=-1).mean()}")


def try_policy(policy, render=True, max_num_steps=200, time_delay=0.1):
    policy.eval()
    env = create_halfcheetah_env(render, 1)
    ob = env.reset()[0]
    steps = 0
    ep_rwds = []
    done = False
    while not done and steps < max_num_steps:
        act = policy.take_np_action(ob)
        if render:
            env.render()
        ob, rwd, done, info, _ = env.step(act)

        ep_rwds.append(rwd)

        steps += 1
        time.sleep(time_delay)

    return np.sum(ep_rwds)

# Example usage
if __name__ == '__main__':
    pass
