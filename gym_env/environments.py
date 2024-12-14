import torch
import numpy as np
from abc import ABC, abstractmethod
import gymnasium as gym

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

class Env(ABC):
    """Interface for gym enviroments """

    def __init__(self, gym_env: gym.Env) -> None:
        super().__init__()
        self.env = gym_env  

    def __getattr__(self, name):
        """
        Delegate method calls and attribute access to the wrapped gym.Env instance.
        """
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @abstractmethod
    def take_step(self, state: np.array, action: np.array):
        """Take Step and Refine Reward"""
        pass

    def get_dim(self):
        return {"states": self.env.observation_space.shape[0], 
                "actions": self.env.action_space.shape[0]}
    
    def get_rollout(self, policy, max_num_steps):

        states = []
        actions = []
        rewards = []

        done = False
        trunc = False

        curr_state = self.env.reset()[0]

        steps = 0
        while not done and not trunc and steps < max_num_steps:
            curr_action = policy.take_np_action(curr_state)
            next_state, curr_reward, done, trunc = self.take_step(curr_state, curr_action)

            states.append(curr_state)
            actions.append(curr_action)
            rewards.append(curr_reward)

            curr_state = next_state
            steps += 1

        return states, actions, rewards
    
    def get_rollout_tensors(self, policy, batch_size, max_num_steps):
        
        # Collect rollouts from the environment
        rollouts = [self.get_rollout(policy, max_num_steps) for _ in range(batch_size)]

        # Unzip the collected rollouts into separate components
        exp_states_b, exp_actions_b, exp_rewards_b = map(lambda x: FloatTensor(x), zip(*rollouts))

        return exp_states_b, exp_actions_b, exp_rewards_b

class HalfCheetahEnv(Env):
    def __init__(self, gym_env: gym.Env) -> None:
        super().__init__(gym_env)

    def take_step(self, state: np.array, action: np.array):
        next_state, reward, done, trunc, _ = self.env.step(action)

        if state[1] > 2.3 or state[1] < -2.3:
            reward -= 100
        elif state[1] > 0.3 or state[1] < -1.5:
            reward -= 10

        return next_state, reward, done, trunc
    

def create_halfcheetah_env(render=False, forward_reward_weight=1):
    render_mode = "human" if render else None
    
    env = gym.make("HalfCheetah-v5", render_mode=render_mode, 
                   forward_reward_weight=forward_reward_weight)
    env = HalfCheetahEnv(env)
    # _ = env.reset(seed=0)
    return env