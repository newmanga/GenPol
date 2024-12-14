from models.diffusion import Diffusion
import numpy as np
import torch
from torch.nn import Module

from models.discriminator import DiscriminatorNet
from models.value import ValueNet
from models.policy import GaussianPolicyNet
from models.trpo import TRPO

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class GAIL(Module, TRPO):
    def __init__(
        self,
        env,
        batch_size,
        enable_diffusion=False,
        config=None
    ) -> None:
        Module.__init__(self)

        self.env = env 
        env_dim = env.get_dim()
        
        self.enable_diffusion = enable_diffusion
        self.batch_size = batch_size
        self.config = config

        self.num_iters              = self.config["num_iters"]
        self.num_steps_per_iter     = self.config["num_steps_per_iter"]
        self.horizon                = self.config["horizon"]
        self.lambda_                = self.config["lambda"]
        self.gae_gamma              = self.config["gae_gamma"]
        self.gae_lambda             = self.config["gae_lambda"]
        self.eps                    = self.config["epsilon"]
        self.max_kl                 = self.config["max_kl"]
        self.cg_damping             = self.config["damping_coeff"]
        self.normalize_advantage    = self.config["normalize_advantage"]
        self.exp_data_refresh_rate  = self.config["exp_ref_rate"]
        self.hidden_sizes           = self.config["hidden_sizes"] 

        policy_net = GaussianPolicyNet(env_dim, self.hidden_sizes)
        value_net = ValueNet(env_dim["states"], self.hidden_sizes)
        TRPO.__init__(self, env_dim["actions"], policy_net, value_net, self.config)

        diffusion = None
        if self.enable_diffusion:
            diffusion = Diffusion(self.config)

        self.discriminator = DiscriminatorNet(env_dim, 
                                              self.config,
                                              diffusion=diffusion)

    def get_log_prob(self, fake_states, fake_actions):
        gen_scores = self.discriminator.get_logits(fake_states, fake_actions)
        # here we would like to get negatives so that policy gets penalized
        return torch.log(torch.sigmoid(gen_scores)).squeeze()#.detach()
    
    def get_advantages_and_returns(self, states, actions, gammas, lambdas):

        log_prob = self.get_log_prob(states, actions)

        returns = self.get_returns(log_prob, gammas)

        deltas = self.get_td_error(states, log_prob.unsqueeze(-1))

        advantages = self.get_advantages(deltas, gammas, lambdas, self.num_steps_per_iter)

        print(f"Generated Prediction: {log_prob.mean()}")

        return advantages, returns
    
    def get_rollout_generator(self):

        # Collect rollouts from the environment
        states_b, actions_b, rewards_b = self.env.get_rollout_tensors(self.policy_net,
                                                                      self.batch_size, 
                                                                      self.num_steps_per_iter)

        # Calculate the discount factors
        gammas, lambdas = map(lambda x: FloatTensor(x), self.get_discounts(len(states_b[0])))

        # Compute the advantages and returns
        advantages, returns = self.get_advantages_and_returns(states_b, actions_b, gammas, lambdas)

        return states_b, actions_b, returns, advantages, gammas, rewards_b
    
    def train(self, expert_policy):

        logs = {"policy/rewards":[], 
                "expert/rewards":[],
                "discriminator/real_logprob":[],
                "discriminator/fake_logprob":[],
                "policy/loss":[],
                "discriminator/real/prob":[]}

        for i in range(self.num_iters):
            print(f"Iteration: {i + 1}")

            if i % self.exp_data_refresh_rate == 0:
                exp_states_b, exp_actions_b, exp_rewards_b = self.env.get_rollout_tensors(expert_policy,
                                                                                          self.batch_size, 
                                                                                          self.num_steps_per_iter)
                logs["expert/rewards"].append(exp_rewards_b.sum(dim=-1).mean())
                print( f" Expert Rewards: {exp_rewards_b.sum(dim=-1).mean()}")

            states, actions, returns, advantages, gammas, rewards = self.get_rollout_generator()
            logs["policy/rewards"].append(rewards.sum(dim=-1).mean())
            print( f" Policy Rewards: {rewards.sum(dim=-1).mean()}")

            real_logprob, fake_logprob = self.discriminator.train_step(exp_states_b, exp_actions_b, states, actions)
            logs["discriminator/real_logprob"].append(real_logprob)
            logs["discriminator/fake_logprob"].append(fake_logprob)
            print(f"Discriminator real_logprob: {real_logprob}")

            self.update_value_net(states, returns)
  
            self.update_policy_net(states, actions, advantages, gammas)

            real_pred = self.discriminator.get_prediction(exp_states_b, exp_actions_b).mean()
            logs["discriminator/real/prob"].append(real_pred)
            print(f"Real Pred: {real_pred}")

            self.discriminator.update_times(real_pred)

        return logs


