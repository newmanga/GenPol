import torch
from torch.nn import Module

from utils_file import *

from models.policy import GaussianPolicyNet
from models.discriminator import DiscriminatorNet
from models.diffusion import Diffusion

class BCGAN(Module):
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
        self.hidden_sizes           = self.config["hidden_sizes"]
        self.num_steps_per_iter     = self.config["num_steps_per_iter"] 

        diffusion = None
        if self.enable_diffusion:
            diffusion = Diffusion(self.config)

        self.policy_net = GaussianPolicyNet(env_dim, self.hidden_sizes)    

        self.discriminator = DiscriminatorNet(env_dim, 
                                              self.config,
                                              diffusion=diffusion)

    def train_policy_step(self, gen_sta):
        self.policy_net.train()
        self.policy_net.optimizer.zero_grad()
        gen_act = self.policy_net.get_dist_t(gen_sta).rsample()
        gen_prob = self.discriminator.get_logits(gen_sta, gen_act).sigmoid()
        loss = -torch.mean(gen_prob)
        loss.backward()
        self.policy_net.optimizer.step()

        return loss.item()

    def train(self, expert_policy):
        logs = {"policy/rewards":[], 
                "expert/rewards":[],
                "discriminator/real_logprob":[],
                "discriminator/fake_logprob":[],
                "policy/loss":[],
                "discriminator/real/prob":[]}

        exp_states_b, exp_actions_b, exp_rewards_b = self.env.get_rollout_tensors(expert_policy,
                                                                            self.batch_size, 
                                                                            self.num_steps_per_iter)
        logs["expert/rewards"].append(exp_rewards_b.sum(dim=-1).mean())
        print( f" Expert Rewards: {exp_rewards_b.sum(dim=-1).mean()}")

        for i in range(self.num_iters):
            print(f"Iteration: {i + 1}")

            gen_states_b, gen_actions_b, gen_rewards_b = self.env.get_rollout_tensors(self.policy_net,
                                                                            self.batch_size, 
                                                                            self.num_steps_per_iter)

            real_logprob, fake_logprob = self.discriminator.train_step(exp_states_b, exp_actions_b, gen_states_b, gen_actions_b)
            logs["discriminator/real_logprob"].append(real_logprob)
            logs["discriminator/fake_real_logprob"].append(fake_logprob)
            print(f"Discriminator real_logprob: {real_logprob}")
            print(f"Discriminator fake_real_logprob: {fake_logprob}")

            gen_fake_logprob = self.train_policy_step(gen_states_b)
            logs["policy/loss"].append(gen_fake_logprob)
            print(f"policy gen_fake_logprob: {gen_fake_logprob}")


            real_pred = self.discriminator.get_prediction(exp_states_b, exp_actions_b).mean()
            logs["discriminator/real/prob"].append(real_pred)
            print(f"Real Pred: {real_pred}")

            self.discriminator.update_times(real_pred)

        return logs