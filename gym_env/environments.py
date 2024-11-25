import torch
from abc import ABC, abstractmethod
import gymnasium as gym

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
    def take_step(self, action):
        """Take Step and Refine Reward"""
        pass

    def get_dim(self):
        return {"states": self.env.observation_space.shape[0], 
                "actions": self.env.action_space.shape[0]}

class HalfCheetahEnv(Env):
    def __init__(self, gym_env: gym.Env) -> None:
        super().__init__(gym_env)

    def take_step(self, state: torch.Tensor, action: torch.Tensor):
        next_state, reward, done, trunc, _ = self.env.step(action.squeeze().numpy())

        if state[0,1] > 2.3 or state[0,1] < -2.3:
            reward -= 100
        elif state[0,1] > 0.3 or state[0,1] < -1.5:
            reward -= 10

        return next_state, reward, done, trunc
    

def create_halfcheetah_env(render=False, forward_reward_weight=1):
    render_mode = "human" if render else None
    
    env = gym.make("HalfCheetah-v5", render_mode=render_mode, 
                   forward_reward_weight=forward_reward_weight)
    env = HalfCheetahEnv(env)
    # _ = env.reset(seed=0)
    return env